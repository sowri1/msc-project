include("./DroneLibV2.jl")
using Serialization

mutable struct Actor
	model::Chain
end

function Actor(state_dim::Int, action_dim::Int, max_action::Float64)
	### Define the parameters of each layer of type Float64
	model = Chain(
		Dense(state_dim, 256, relu; init =Flux.glorot_uniform(gain=2)),
		BatchNorm(256),
		Dense(256, 512, relu; init =Flux.glorot_uniform(gain=2)),
		BatchNorm(512),
		Dense(512, 1024, relu; init =Flux.glorot_uniform(gain=2)),
		BatchNorm(1024),
		Dense(1024, 1024, relu; init =Flux.glorot_uniform(gain=2)),
		BatchNorm(1024),
		Dense(1024, 256, relu; init =Flux.glorot_uniform(gain=2)),
		BatchNorm(256),
		Dense(256, 64, relu; init =Flux.glorot_uniform(gain=2)),
		BatchNorm(64),
		Dense(64, action_dim, sigmoid; init =Flux.glorot_uniform(gain=2))
	) |> gpu
	model = fmap(f64, model)
	return Actor(model)
end

mutable struct Critic
	model::Chain
end

function Critic(state_dim::Int, action_dim::Int)
	model = Chain(
		Dense(state_dim + action_dim, 256; init =Flux.glorot_uniform(gain=2)),
		BatchNorm(256),
		Dense(256, 1024; init =Flux.glorot_uniform(gain=2)),
		BatchNorm(1024),
		Dense(1024, 256; init =Flux.glorot_uniform(gain=2)),
		BatchNorm(256),
		Dense(256, 128;  init =Flux.glorot_uniform(gain=2)),
		BatchNorm(128),
		Dense(128, 1; init =Flux.glorot_uniform(gain=2))
	) |> gpu
	model = fmap(f64, model)
	return Critic(model)
end

struct DDPGAgent
	actor::Actor
	target_actor::Actor
	critic::Critic
	target_critic::Critic
	actor_optimizer
	critic_optimizer
	gamma::Float64
	tau::Float64
	max_action::Float64
end

function DDPGAgent(state_dim::Int, action_dim::Int, max_action::Float64;
		actor_lr::Float64=0.001, critic_lr::Float64=0.002, gamma::Float64=0.99, tau::Float64=0.05)

	actor = Actor(state_dim, action_dim, max_action)
	target_actor = Actor(state_dim, action_dim, max_action)
	Flux.loadparams!(target_actor.model, Flux.params(actor.model))

	critic = Critic(state_dim, action_dim)
	target_critic = Critic(state_dim, action_dim)
	Flux.loadparams!(target_critic.model, Flux.params(critic.model))

	actor_optimizer = ADAM(actor_lr)
	critic_optimizer = ADAM(critic_lr)

	return DDPGAgent(actor, target_actor, critic, target_critic, actor_optimizer, critic_optimizer, gamma, tau, max_action)
end

function DDPGAgent(state_dim::Int, action_dim::Int, max_action::Float64, critic_mode_file::String, actor_mode_file::String;
	actor_lr::Float64=0.001, critic_lr::Float64=0.002, gamma::Float64=0.99, tau::Float64=0.005)

	@load actor_mode_file actor_model
	@load critic_mode_file critic_model

	actor = Actor(state_dim, action_dim, max_action)
	Flux.loadmodel!(actor.model, actor_model)
	target_actor = Actor(state_dim, action_dim, max_action)
	Flux.loadparams!(target_actor.model, Flux.params(actor.model))

	critic = Critic(state_dim, action_dim)
	Flux.loadmodel!(critic.model, critic_model)
	target_critic = Critic(state_dim, action_dim)
	Flux.loadparams!(target_critic.model, Flux.params(critic.model))

	actor_optimizer = ADAM(actor_lr)
	critic_optimizer = ADAM(critic_lr)

	return DDPGAgent(actor, target_actor, critic, target_critic, actor_optimizer, critic_optimizer, gamma, tau, max_action)
end

function save_model(agent::DDPGAgent, actor_file::String = "agent_actor_model.bson", critic_file::String="agent_critic_model.bson")
	actor_model = Flux.cpu(agent.target_actor.model);
	@save actor_file actor_model
	
	critic_model = Flux.cpu(agent.target_critic.model);
	@save critic_file critic_model
end

function soft_update!(target::Chain, source::Chain, tau::Float64)
	for (target_param, source_param) in zip(Flux.params(target), Flux.params(source))
		target_param .= (1 - tau) * target_param .+ tau * source_param
	end
end

### Implement noise
mutable struct OUNoise
	noise::CuArray{Float64}
	theta::Float64
	mu::Float64
	sigma::Float64
end

function OUNoise(action_dim::Int, theta::Float64=0.15, mu::Float64=0.0, sigma::Float64=0.2)
	noise = zeros(action_dim) |> gpu
	return OUNoise(noise, theta, mu, sigma)
end

### Generate noise
function generateNoise!(noise::OUNoise)
	noise.noise .+= ( noise.theta * (noise.mu .- noise.noise) * (1 / frequency) ) + ( noise.sigma * (1 / frequency) * randn!(noise.noise) )
	noise.noise = noise.noise |> gpu
	return noise.noise
end

function genGPUNoise(stddev::Float64, action_dim::Tuple{Int64})
	return CuArray{Float64}(randn(action_dim) .* stddev)
end

function genGPUNoise(stddev::Float64, action_dim::Tuple{Int64, Int64})
	return CuArray{Float64}(randn(action_dim) .* stddev)
end

function addParamterNoise!(agent::DDPGAgent, noise_std::Float64=0.001)
	new_params = deepcopy(Flux.params(agent.actor.model))
	new_params = map(p -> p .+ genGPUNoise(noise_std, size(p)), new_params)
	Flux.loadparams!(agent.actor.model, new_params)
end

function get_action(agent::DDPGAgent, state::CuArray{Float64}, gen_rand::Bool=false, noise_factor::Float64=1.0, rand_factor::Float64=0.1)
	# state = reshape(state, (:, 1))
	# input_state = vcat(state, previous_action, target_position) |> gpu
	input_state = reshape(state, (:, 1))
	action = agent.actor.model(input_state)
	add_noise_sequence = [
		[1, 2],
		[1, 4],
		[2, 3],
		[3, 4],
	]
	rand_noise = (ones(4) .* abs(rand())) |> gpu
	action .+= rand_noise * (gen_rand && rand() < noise_factor)
	for (m1, m2) in add_noise_sequence
		if (gen_rand && rand() < noise_factor)
			if rand() > 0.5
				action[m1] += action[m1] * 2e-3 * rand()
				action[m2] += action[m2] * 2e-3 * rand()
			else
				action[m1] -= action[m1] * 2e-3 * rand()
				action[m2] -= action[m2] * 2e-3 * rand()
			end
		end
	end
	# if (gen_rand && rand() < noise_factor)
	# 	action .+= generateNoise!(noise)
	# end
	action = clamp.(action, 0, 1)
	return convert(CuArray{Float64}, reshape(action, :))
end

function update(agent::DDPGAgent, batch_states::Matrix{Float64}, batch_actions::Matrix{Float64},
	batch_next_states::Matrix{Float64}, batch_rewards::Vector{Float64}, batch_dones::Vector{Bool})

	batch_states = batch_states |> gpu
	batch_actions = batch_actions |> gpu
	batch_next_states = batch_next_states |> gpu
	batch_rewards = batch_rewards |> gpu
	batch_dones = batch_dones |> gpu

	# Reshape the batch rewards and dones
	batch_rewards = reshape(batch_rewards, (1, :))
	batch_dones = reshape(batch_dones, (1, :))

	# Calculate the target Q value
	target_actions = agent.target_critic.model(vcat(batch_next_states, agent.target_actor.model(batch_next_states)))  |> gpu
	target_q_values = (batch_rewards .+ agent.gamma .* (1 .- batch_dones) .* target_actions)  |> gpu

	critic_optim = Flux.setup(agent.critic_optimizer, agent.critic.model)
	actor_optim = Flux.setup(agent.actor_optimizer, agent.actor.model)

	# Update the critic
	gs = Flux.gradient(agent.critic.model) do m
		return Flux.mse(m(vcat(batch_states, batch_actions)), target_q_values)
	end
	critic_optim, agent.critic.model = Flux.update!(critic_optim, agent.critic.model, gs[1])

	# update!(agent.critic_optimizer, m, gs)

	# Update the actor
	gs = Flux.gradient(agent.actor.model) do m
		return -mean(agent.critic.model(vcat(batch_states, m(batch_states))))
	end
	actor_optim, agent.actor.model = Flux.update!(actor_optim, agent.actor.model, gs[1])

end

### Implement replay buffer
struct ReplayBuffer
	buffer::Vector{Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64, Bool}}
	capacity::Int
end

function ReplayBuffer(state_dim::Int, action_dim::Int, capacity::Int)
	buffer = Vector{Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64, Bool}}()
	return ReplayBuffer(buffer, capacity)
end

function bufferPush!(buffer::ReplayBuffer, state::CuArray{Float64}, action::CuArray{Float64}, next_state::CuArray{Float64}, reward::Float64, done::Bool)
	if length(buffer.buffer) == buffer.capacity
		popfirst!(buffer.buffer)
	end
	push!(buffer.buffer, (state, action, next_state, reward, done))
end

function sample!(buffer::ReplayBuffer, batch_size::Int)
	batch = sample(buffer.buffer, batch_size, replace=false)
	batch_states = Matrix{Float64}(undef, length(batch[1][1]), 0)
	batch_actions = Matrix{Float64}(undef, length(batch[1][2]), 0)
	batch_next_states = Matrix{Float64}(undef, length(batch[1][3]), 0)
	batch_rewards = Vector{Float64}(undef, 0)
	batch_dones = Vector{Bool}(undef, 0)
	for (state, action, next_state, reward, done) in batch
		batch_states = hcat(batch_states, state)
		batch_actions = hcat(batch_actions, action)
		batch_next_states = hcat(batch_next_states, next_state)
		push!(batch_rewards, reward)
		push!(batch_dones, done)
	end
	return batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones
end

### Implement evaluation process
function evaluate(agent::DDPGAgent, env::DroneEnv, max_episodes::Int)
	for episode in 1:max_episodes
		reset!(env)
		episode_reward = 0.0
		while true
			state = get_state(env.drone)
			action = get_action(agent, env.action, env.target_position, state)
			reward, done = step!(env, action.* agent.max_action, true)
			episode_reward += reward
			if done
				break
			end
		end
		println("Episode: $episode, Episodic Reward: $episode_reward. Position: $(env.drone.position)")
	end
end

function get_normalised_state(st, pv_act, tpos)
	state = convert(CuArray{Float64}, deepcopy(st))
	target_position = convert(CuArray{Float64}, deepcopy(tpos))
	prev_action = convert(CuArray{Float64}, deepcopy(pv_act))

	state[1] = state[1] / BOUND_MAX
	state[2] = state[2] / BOUND_MAX
	state[3] = state[3] / BOUND_MAX

	state[7] = sin(state[7])
	state[8] = sin(state[8])
	state[9] = sin(state[9])

	target_position ./= BOUND_MAX
	prev_action ./= MAX_ACTION

	return vcat(state, prev_action, target_position)
end
### Implement training process
function train(agent::DDPGAgent, buffer::ReplayBuffer, batch_size::Int, max_episodes::Int, max_time::Float64=10.0, exploration_noise::Float64=0.5, std_dev::Float64=1e-3)
	### Keep track of the rewards
	rewards_steps = []

	for episode in 1:max_episodes
		target_position = [rand(-10:10), rand(-10:10), rand(1:10)]
		target_position = convert(Array{Float64, 1}, target_position)
		env = DroneEnv(target_position, max_time)
		episode_reward = 0.0
		num_steps = 0

		### Every 10% of episodes done decrease the std_dev
		if episode % (max_episodes / 10) == 0
			# std_dev -= std_dev / 10
			std_dev /= 2
		end

		# addParamterNoise!(agent, std_dev)
		while true
			state = get_state(env.drone)
			# action = get_action(agent, env.action, env.target_position, state)
			add_state = get_normalised_state(state, env.action, env.target_position)
			action = get_action(agent, add_state, true, exploration_noise * (1 - episode / max_episodes))

			reward, done = step!(env, (action .* agent.max_action))
			next_state = get_state(env.drone)

			add_next_state = get_normalised_state(next_state, action, env.target_position)

			bufferPush!(buffer, add_state, action, add_next_state, reward, done)

			### Delayed training
			if length(buffer.buffer) > batch_size
				batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = sample!(buffer, batch_size)
				update(agent, batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones)
			end

			num_steps += 1
			episode_reward += reward

			if done
				break
			end
		end
		push!(rewards_steps, [episode_reward, num_steps])

		# Soft update target networks
		soft_update!(agent.target_actor.model, agent.actor.model, agent.tau)
		soft_update!(agent.target_critic.model, agent.critic.model, agent.tau)

		println("Episode: $episode, Episodic Reward: $episode_reward. Total steps: $num_steps")
		if episode % 10 == 0
			# if episode % 10 == 0
			save_model(agent, "agent_actor_model_1.bson", "agent_critic_model_1.bson")
			### Save training buffer
			serialize("training-buffer_1.dat", buffer)
			# end
		end
	end
	return rewards_steps
end

function run_model(agent::DDPGAgent, env::DroneEnv, gen_rand::Bool=false)
	motors_rpm = []
	state_spaces = []
	reset!(env)
	episode_reward = 0.0
	num_steps = 0
	while true
		state = get_state(env.drone)
		add_state = get_normalised_state(state, env.action, env.target_position)
		action = get_action(agent, add_state, false)
		# action = get_action(agent, env.action, env.target_position, state, gen_rand, 1.0)
		reward, done = step!(env, action.* agent.max_action, true)
		episode_reward += reward
		push!(motors_rpm, convert(Array{Float64, 1}, action))
		push!(state_spaces, convert(Array{Float64, 1}, state))
		num_steps += 1
		if done
			break
		end
	end
	println("Episode reward: ", episode_reward, " Number of steps: ", num_steps)
	return state_spaces, motors_rpm
end