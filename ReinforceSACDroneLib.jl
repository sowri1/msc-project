include("./DroneLibV2.jl")
include("./ReplayBuffer.jl")

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
		Dense(64, action_dim * 2, tanh; init =Flux.glorot_uniform(gain=2))
	) |> gpu
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
		Dense(256, 128; init =Flux.glorot_uniform(gain=2)),
		BatchNorm(128),
		Dense(128, 1; init =Flux.glorot_uniform(gain=2))
	) |> gpu
	return Critic(model)
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

function GenereateNoise(noise::OUNoise)
	noise.noise .+= noise.theta .* (noise.mu .- noise.noise) .+ noise.sigma .* randn!(noise.noise)
	return noise.noise
end

struct SACAgent
	actor::Actor
	target_actor::Actor
	q1::Critic
	target_q1::Critic
	q2::Critic
	target_q2::Critic
	actor_optimizer
	q1_optimizer
	q2_optimizer
	log_alpha::Float64
	alpha_optimizer
	alpha::Float64
	target_entropy::Float64
	gamma::Float64
	tau::Float64
	max_action::Float64
	noise_dist::OUNoise
end

function SACAgent(state_dim::Int, action_dim::Int, max_action::Float64;
		actor_lr::Float64=-0.001, critic_lr::Float64=0.002, gamma::Float64=0.99, tau::Float64=0.05, log_alpha::Float64=1.0)

	actor = Actor(state_dim, action_dim, max_action)
	target_actor = Actor(state_dim, action_dim, max_action)
	Flux.loadparams!(target_actor.model, Flux.params(actor.model))

	q1 = Critic(state_dim, action_dim)
	target_q1 = Critic(state_dim, action_dim)
	Flux.loadparams!(target_q1.model, Flux.params(q1.model))

	q2 = Critic(state_dim, action_dim)
	target_q2 = Critic(state_dim, action_dim)
	Flux.loadparams!(target_q2.model, Flux.params(q2.model))

	actor_optimizer = ADAM(actor_lr)
	q1_optimizer = ADAM(critic_lr)
	q2_optimizer = ADAM(critic_lr)

	alpha = exp(log_alpha)
	target_entropy = -action_dim
	noise_dist = OUNoise(action_dim)

	alpha_optimizer = ADAM(0.0003)

	return SACAgent(actor, target_actor, q1, target_q1, q2, target_q2, actor_optimizer, q1_optimizer, q2_optimizer, log_alpha, alpha_optimizer, alpha, target_entropy, gamma, tau, max_action, noise_dist)
end

function differentiable_min(x::CuArray{Float32}, y::CuArray{Float32})
	return x .+ (y .- x) .* (x .> y)
end


# function save_model(agent::DDPGAgent, actor_file::String = "agent_actor_model.bson", critic_file::String="agent_critic_model.bson")
# 	actor_model = Flux.cpu(agent.target_actor.model);
# 	@save actor_file actor_model
	
# 	critic_model = Flux.cpu(agent.target_critic.model);
# 	@save critic_file critic_model
# end

function soft_update!(target::Chain, source::Chain, tau::Float64)
	for (target_param, source_param) in zip(Flux.params(target), Flux.params(source))
		target_param .= (1 - tau) * target_param .+ tau * source_param
	end
end

function get_action(agent::SACAgent, previous_action::Vector{Float64}, target_position::Vector{Float64}, state::CuArray{Float32, 1}, gen_rand::Bool=false, rand_factor::Float64=0.1)
	pass_state = vcat(state, previous_action, target_position)
	pass_state = pass_state |> gpu
	pass_state = reshape(pass_state, (:, 1))
	output = agent.actor.model(pass_state) |> cpu
	mean_action = output[1:4]
	log_std = output[5:8]
	std = exp.(log_std)
	act_dist = MvNormal(mean_action, std)
	xt = rand(act_dist)
	yt = tanh.(xt)
	action = (yt .* -0.5) .+ 0.5
	log_prob = logpdf(act_dist, xt)
	log_prob -= sum(log.(1 .- yt.^2))
	mean = (tanh.(mean_action) .* -0.5) .+ 0.5
	return action |> gpu, log_prob |> gpu, mean |> gpu
end

function get_action_batch(model::Chain, previous_action::CuArray{Float32, 2}, target_position::CuArray{Float32, 2}, state::CuArray{Float32, 2}, gen_rand::Bool=false, rand_factor::Float64=0.1)
	output = model(state) |> cpu
	mean_action = output[1:4, :]
	log_std = output[5:8, :]
	std = exp.(log_std)
	act_dist = [ MvNormal(mean_action[:, i], std[:, i]) for i in 1:size(mean_action)[2] ]
	xt = [ rand(act_dist[i]) for i in 1:size(act_dist)[1] ]
	### Convert xt to matrix for easier processing
	xt = hcat(xt...)
	xt = reshape(xt, (size(act_dist)[1], :))
	yt = tanh.(xt)
	### Convert yt to matrix for easier processing
	action = (yt .* -0.5) .+ 0.5
	log_prob = [logpdf(act_dist[i], xt[i, :]) for i in 1:size(act_dist)[1]] .- sum(log.(1 .- yt.^2 .+ 1e-8))
	mean = (tanh.(mean_action) .* -0.5) .+ 0.5
	### Convert action, log_prob, mean to matrix for easier processing and transpose
	action = reshape(action, (:, size(action)[1]))
	log_prob = reshape(log_prob, (:, size(log_prob)[1]))
	mean = reshape(mean, (size(mean)[1], :))
	return action |> gpu, log_prob |> gpu, mean |> gpu

end

function update(agent::SACAgent, batch_states::Matrix{Float64}, batch_actions::Matrix{Float64},
	batch_next_states::Matrix{Float64}, batch_rewards::Vector{Float64}, batch_dones::Vector{Bool})

	batch_states = batch_states |> gpu
	batch_actions = batch_actions |> gpu
	batch_next_states = batch_next_states |> gpu
	batch_rewards = batch_rewards |> gpu
	batch_rewards = reshape(batch_rewards, (1, :))
	batch_dones = batch_dones |> gpu
	batch_dones = reshape(batch_dones, (1, :))

	next_state_action, next_state_log_prob, next_state_mean = get_action_batch(agent.actor.model, batch_actions, batch_states, batch_next_states, false)
	qf1_next_target = agent.target_q1.model(vcat(batch_next_states, next_state_action))
	qf2_next_target = agent.target_q2.model(vcat(batch_next_states, next_state_action))
	min_qf_next_target = differentiable_min(qf1_next_target, qf2_next_target) - agent.alpha * next_state_log_prob
	next_q_value = batch_rewards .+ (1 .- batch_dones) .* agent.gamma .* min_qf_next_target

	q1_optim_state = Flux.setup(agent.q1_optimizer, agent.q1.model)
	q2_optim_state = Flux.setup(agent.q2_optimizer, agent.q2.model)
	actor_optim_state = Flux.setup(agent.actor_optimizer, agent.actor.model)
	alpha_optim_state = Flux.setup(agent.alpha_optimizer, agent.alpha)

	
	for i in 1:EPOCHS
		q1_grad = Flux.gradient(agent.q1.model) do m
			return Flux.mse(m(vcat(batch_states, batch_actions)), next_q_value)
		end
		Flux.update!(q1_optim_state, agent.q1.model, q1_grad[1])
		# q1_optim_state, agent.q1.model = Flux.update!(q1_optim_state, agent.q1.model, q1_grad[1])
	
		q2_grad = Flux.gradient(agent.q2.model) do m
			return Flux.mse(m(vcat(batch_states, batch_actions)), next_q_value)
		end
		Flux.update!(q2_optim_state, agent.q2.model, q2_grad[1])
		# q2_optim_state, agent.q2.model = Flux.update!(q2_optim_state, agent.q2.model, q2_grad[1])

		alpha_grad = Flux.gradient(agent.log_alpha) do m
			return -mean(agent.log_alpha .* (next_state_log_prob .+ agent.target_entropy))
		end
		# Flux.update!(alpha_optim_state, agent.log_alpha, alpha_grad[1])
		alpha_optim_state, agent.log_alpha = Flux.update!(alpha_optim_state, agent.log_alpha, alpha_grad[1])
		agent.alpha = exp(agent.log_alpha)

		actor_grad = Flux.gradient(agent.actor.model) do m
			pi, log_pi, _ = get_action_batch(m, batch_actions, batch_states, batch_next_states, false)
			qf1_pi = agent.q1.model(vcat(batch_states, pi))
			qf2_pi = agent.q2.model(vcat(batch_states, pi))
			min_qf_pi = differentiable_min(qf1_pi, qf2_pi)
			actor_loss = mean(agent.alpha .* log_pi .- min_qf_pi)
			return actor_loss
		end
		println(actor_grad[1])
		Flux.update!(actor_optim_state, agent.actor.model, actor_grad[1])	
		actor_optim_state, agent.actor.model = Flux.update!(actor_optim_state, agent.actor.model, actor_grad[1])	
	end

	soft_update!(agent.target_actor.model, agent.actor.model, agent.tau)
	soft_update!(agent.target_q1.model, agent.q1.model, agent.tau)
	soft_update!(agent.target_q2.model, agent.q2.model, agent.tau)
end

### Implement evaluation process
function evaluate(agent::SACAgent, env::DroneEnv, max_episodes::Int)
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

### Implement training process
function train(agent::SACAgent, buffer::ReplayBuffer, batch_size::Int, max_episodes::Int, max_time::Float64=10.0, exploration_noise::Float64=0.5, std_dev::Float64=1e-3)
	### Keep track of the rewards
	rewards_steps = []

	for episode in 1:max_episodes
		target_position = [rand(-10:10), rand(-10:10), rand(1:10)]
		target_position = convert(Array{Float64, 1}, target_position)
		target_pos_gpu = target_position |> gpu
		target_pos_gpu = reshape(target_pos_gpu, (:, 1))
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

			action, _, __ = get_action(agent, env.action, env.target_position, state, true, exploration_noise * (1 - episode / max_episodes))

			reward, done = step!(env, (action .* agent.max_action))
			next_state = get_state(env.drone)

			add_state = vcat(state, action, env.target_position)
			add_next_state = vcat(next_state, action, env.target_position)
			action = convert(CuArray{Float64, 1}, action)

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

		if episode % 100 == 0
			println("Episode: $episode, Episodic Reward: $episode_reward. Total steps: $num_steps")
			if episode % 1000 == 0
				save_model(agent, "agent_actor_model_1.bson", "agent_critic_model_1.bson")
				### Save training buffer
				serialize("training-buffer_1.dat", buffer)
			end
		end
	end
	return rewards_steps
end

function run_model(agent::SACAgent, env::DroneEnv, gen_rand::Bool=false)
	motors_rpm = []
	state_spaces = []
	reset!(env)
	episode_reward = 0.0
	num_steps = 0
	while true
		state = get_state(env.drone)
		action, _, __ = get_action(agent, env.action, env.target_position, state, gen_rand, 1.0)
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