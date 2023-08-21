include("./DroneLibV3.jl")

# Define the PPO agent
mutable struct PPOAgent
	policy_net::Chain
	value_net::Chain

	# target_policy_net::Chain
end

function PPOAgent(input_dim::Int, output_dim::Int)
	### LSTM Network
	# policy_net = Chain(
	# 	LSTM(input_dim, 128),
	# 	# LSTM(128, 256),
	# 	Dense(128, 256, relu, init=Flux.glorot_uniform()),
	# 	Dense(256, 512, relu, init=Flux.glorot_uniform()),
	# 	Dense(512, 1024, relu, init=Flux.glorot_uniform()),
	# 	Dense(1024, 1024, relu, init=Flux.glorot_uniform()),
	# 	Dense(1024, 512, relu, init=Flux.glorot_uniform()),
	# 	Dense(512, 256, relu, init=Flux.glorot_uniform()),
	# 	Dense(256, 128, relu, init=Flux.glorot_uniform()),
	# 	Dense(128, output_dim, init=Flux.glorot_uniform()),
	# ) |> gpu
	policy_net = Chain(
		Dense(input_dim, 128, relu, init=Flux.glorot_uniform()),
		BatchNorm(128),
		Dense(128, 256, relu, init=Flux.glorot_uniform()),
		# BatchNorm(256),
		Dense(256, 1024, relu, init=Flux.glorot_uniform()),
		# BatchNorm(1024),
		Dense(1024, 1024, relu, init=Flux.glorot_uniform()),
		# BatchNorm(1024),
		Dense(1024, 128, relu, init=Flux.glorot_uniform()),
		BatchNorm(128),
		Dense(128, output_dim, sigmoid, init=Flux.glorot_uniform()),
	) |> gpu

	value_net = Chain(
		Dense(input_dim, 128, relu, init=Flux.glorot_uniform()),
		# BatchNorm(128),
		Dense(128, 128, relu, init=Flux.glorot_uniform()),
		# BatchNorm(128),
		Dense(128, 1, init=Flux.glorot_uniform()),
	) |> gpu

	policy_net = fmap(f64, policy_net)
	value_net = fmap(f64, value_net)

	# target_policy_net = deepcopy(policy_net)

	return PPOAgent(policy_net, value_net)
end

function PPOAgent(input_dim::Int, output_dim::Int, actor_mode_file::String, critic_mode_file::String)
	agent = PPOAgent(input_dim, output_dim)

	@load actor_mode_file actor_model
	@load critic_mode_file critic_model

	actor_model = fmap(f64, actor_model)
	critic_model = fmap(f64, critic_model)

	Flux.loadmodel!(agent.policy_net, actor_model)
	# Flux.loadmodel!(agent.target_policy_net, actor_model)
	Flux.loadmodel!(agent.value_net, critic_model)

	return agent
end

function softcopy_model_params(target::Chain, source::Chain, tau::Float64=0.05)
	for (target_param, source_param) in zip(Flux.params(target), Flux.params(source))
		target_param .= (1 - tau) * target_param .+ tau * source_param
	end
end

function guassian_likelihood(x, mu)
	pre_sum = -log.((sqrt.(2 .* pi .* (STD.^2))) .+ 1e-8) .- ((x .- mu).^2 ./ ((2 .* STD.^2) .+ 1e-8))
	gl = sum(pre_sum, dims=2)
	if any(isnan.(gl))
		# println("X: ", x, " Mu: ", mu, " Pre Sum: ", pre_sum, " GL: ", gl)
		throw(DomainError("NaN in guassian likelihood"))
	end
	return gl
end

# PPO loss functions
function compute_policy_loss(m::Chain, states, actions, advantages, old_log_probs)
	outputs = m(states)

	new_log_probs = guassian_likelihood(actions, outputs)
	### Check if any of the new log probs are NaN
	if any(isnan.(new_log_probs))
		if any(isnan.(outputs))
			# println("Outputs: ", outputs)
			throw(DomainError("NaN in outputs"))
		end
		# println("Actions: ", actions, " Outputs: ", outputs, " New Log Probs: ", new_log_probs, " Old Log Probs: ", old_log_probs)
		throw(DomainError("NaN in new log probs"))
	end

	# Calculate the entropy of the policy
	# entrp = entropy.(distributions)

	# Compute the ratio of probabilities and the clipped advantages
	ratio = exp.(new_log_probs .- old_log_probs)

	p1 = ratio .* advantages
	p2 = clamp.(ratio, 1 - EPSILON_CLIP, 1 + EPSILON_CLIP) .* advantages

	p3 = p1 .- p2
	p3 = p3 .^ 2
	p3 = sqrt.(p3)

	loss = -mean((p1 .+ p2 .- p3) ./ 2)

	if any(isnan.(loss))
		# println("Actions: ", actions, " Outputs: ", outputs, " New Log Probs: ", new_log_probs, " Old Log Probs: ", old_log_probs)
		# println("P1: ", p1, " P2: ", p2, " P3: ", p3, " Loss: ", loss)
		throw(DomainError("NaN in loss"))
	end

	return loss
end

function compute_value_loss(m::Chain, states, returns)
	values = m(states)
	values = reshape(values, size(returns))
	loss = Flux.mse(values, returns)
	return loss
end

function discount_rewards(rewards)
	discounted_rewards = zeros(size(rewards))
	next_reward = 0.0
	for t in reverse(1:length(rewards))
		reward = rewards[t]
		next_reward = reward + GAMMA * next_reward
		discounted_rewards[t] = next_reward
	end

	return discounted_rewards
end

function get_gaes(rewards, dones, values, next_values, normalize::Bool=true)
	deltas = rewards .+ GAMMA * next_values .* (1 .- dones) .- values

	gaes = deepcopy(deltas)
	CUDA.allowscalar() do
		for t in reverse(1:length(deltas) - 1)
			gaes[t] = gaes[t] + GAMMA * LAMBDA * gaes[t + 1] * (1 - dones[t])
		end
	end

	target = gaes .+ values
	if normalize
		normalize!(target)
	end
	return gaes, target
end

function replay(agent::PPOAgent, states, actions, rewards, dones, next_states, logp_ts)
	values = agent.value_net(states)
	next_values = agent.value_net(next_states)

	advantages, targets = get_gaes(rewards, dones, values, next_values)

	policy_optim = Flux.setup(Flux.Adam(1e-4), agent.policy_net)
	value_optim = Flux.setup(Flux.Adam(2e-3), agent.value_net)

	# train_set = Flux.Data.DataLoader((states, actions, advantages, logp_ts, targets), batchsize=min(BATCH_SIZE, size(states)[1]), shuffle=true)

	for i in 1:EPOCHS
		# for (bt_states, bt_actions, bt_advantages, bt_logp_ts, bt_targets) in train_set
		# 	policy_grads = Flux.gradient(agent.policy_net) do m
		# 		return compute_policy_loss(m, bt_states, bt_actions, bt_advantages, bt_logp_ts)
		# 	end
		# 	policy_optim, agent.policy_net = Flux.update!(policy_optim, agent.policy_net, policy_grads[1])
		# 	policy_grads = nothing

		# 	value_grads = Flux.gradient(agent.value_net) do m
		# 		return compute_value_loss(m, bt_states, bt_targets)
		# 	end
		# 	value_optim, agent.value_net = Flux.update!(value_optim, agent.value_net, value_grads[1])
		# 	value_grads = nothing
		# end
		# bt_states, bt_actions, bt_advantages, bt_logp_ts, bt_targets = nothing, nothing, nothing, nothing, nothing

		policy_grads = Flux.gradient(agent.policy_net) do m
			return compute_policy_loss(m, states, actions, advantages, logp_ts)
		end
		policy_optim, agent.policy_net = Flux.update!(policy_optim, agent.policy_net, policy_grads[1])
		policy_grads = nothing

		value_grads = Flux.gradient(agent.value_net) do m
			return compute_value_loss(m, states, targets)
		end
		value_optim, agent.value_net = Flux.update!(value_optim, agent.value_net, value_grads[1])
		value_grads = nothing
	end
	policy_optim = nothing
	value_optim = nothing
	advantages = nothing
	targets = nothing
	log_std_grads = nothing
	values = nothing
	next_values = nothing
	# train_set = nothing
end

function get_normalised_state(st, tpos, time)
	state = deepcopy(st)
	target_position = deepcopy(tpos)

	state[1] = state[1] / BOUND_MAX
	state[2] = state[2] / BOUND_MAX
	state[3] = state[3] / BOUND_MAX

	state[10] = sin(state[10])
	state[11] = sin(state[11])
	state[12] = sin(state[12])

	# ###Scale the angular velocities by a factor of 1e10
	# state[10] = state[10] / 1e10
	# state[11] = state[10] / 1e10
	# state[12] = state[10] / 1e10

	target_position ./= BOUND_MAX

	return vcat(state, target_position, time)
	# return vcat(st, tpos, time)
end

function train_drone_env(agent::PPOAgent, num_episodes::Int, max_time::Float64=2.0)
	states, actions, rewards, next_states, old_log_probs, dones = CuArray{Float64}(undef, 22, 0), CuArray{Float64}(undef, 4, 0), CuArray{Float64}(undef, 1, 0), CuArray{Float64}(undef, 22, 0), CuArray{Float64}(undef, 4, 0), CuArray{Bool}(undef, 1, 0)

	for episode in 1:num_episodes
		target_position = [0, 0, rand(1:10)]
		# target_position = [rand(-10:10), rand(-10:10), rand(1:10)]
		target_position = convert(Array{Float64, 1}, target_position)
		env = DroneEnv(target_position, max_time)
		reset!(env)

		done = false
		total_reward = 0.0
		reset!(env)
		num_steps = 0
		while !done
			state = get_state(env.drone)
			add_state = get_normalised_state(state, target_position, env.time)

			action, log_prob = sample_action(agent, add_state, (1 - episode / num_episodes))

			states = hcat(states, add_state)
			actions = hcat(actions, action)
			old_log_probs = hcat(old_log_probs, log_prob)

			reward, done = step!(env, action .* MAX_ACTION)

			next_state = get_state(env.drone)
			add_next_state = vcat(next_state, target_position, env.time)
			add_next_state = reshape(add_next_state, (:, 1))

			next_states = hcat(next_states, add_next_state)
			dones = hcat(dones, done)
			rewards = hcat(rewards, reward)

			total_reward += reward
			num_steps += 1
		end

		replay(agent, states, actions, rewards, dones, next_states, old_log_probs)
		if episode % 10 == 0
			println("Episode: $episode, Total Reward: $(round(total_reward, digits=2)), Average Reward: $(round(total_reward / num_steps, digits=2)), Steps: $num_steps")
			save_model(agent, "ppo-drone-model")
		end
	
		# softcopy_model_params(agent.target_policy_net, agent.policy_net)
	
		### Empty the arrays
		states = CuArray{Float64}(undef, 22, 0)
		actions = CuArray{Float64}(undef, 4, 0)
		rewards = CuArray{Float64}(undef, 1, 0)
		next_states = CuArray{Float64}(undef, 22, 0)
		old_log_probs = CuArray{Float64}(undef, 4, 0)
		dones = CuArray{Bool}(undef, 1, 0)
	end
end

function sample_action(agent::PPOAgent, state, exploration_noise::Float64=0.0)
	# state = reshape(state, size(state)[1])
	state = reshape(state, (:, 1))
	pred = agent.policy_net(state)

	action = deepcopy(pred)
	action = pred + ((rand(MvNormal(GAUSSIAN_MEAN_CPU, STD_CPU))) |> gpu)
	action = clamp.(action, 0.0, 1.0)
	logp_t = guassian_likelihood(action, pred)

	action = reshape(action, OUTPUT_DIM)
	return action, logp_t
end

function run_model(agent::PPOAgent, target_position::Array{Float64, 1}, max_time::Float64=10.0)
	env = DroneEnv(target_position, max_time)
	motors_rpm = []
	state_spaces = []

	done = false
	total_reward = 0.0
	reset!(env)
	num_steps = 0
	while !done
		state = get_state(env.drone)
		add_state = get_normalised_state(state, target_position, env.time)

		add_state = reshape(add_state, (:, 1))
		action = agent.policy_net(add_state)
		action = reshape(action, OUTPUT_DIM)

		reward, done = step!(env, action .* MAX_ACTION, true)

		push!(motors_rpm, convert(Array{Float64, 1}, action))
		push!(state_spaces, convert(Array{Float64, 1}, state))

		total_reward += reward
		num_steps += 1
	end
	println("Total Reward: $total_reward", " Target Position: $target_position", " Final Position: $(env.drone.position), Num Steps: $num_steps")
	return state_spaces, motors_rpm
end

function save_model(agent::PPOAgent, path::String)
	for ps in Flux.params(agent.policy_net)
		if any(isnan.(ps))
			println("NaN in parameters")
			return
		end
	end
	for ps in Flux.params(agent.value_net)
		if any(isnan.(ps))
			println("NaN in parameters")
			return
		end
	end
	actor_model = Flux.cpu(agent.policy_net);
	@save "$path/actor_model_ppo.bson" actor_model
	# actor_model = Flux.cpu(agent.target_policy_net);
	# @save "$path/actor_model_ppo.bson" actor_model

	critic_model = Flux.cpu(agent.value_net);
	@save "$path/critic_model_ppo.bson" critic_model
end