include("./DroneLibV2.jl")

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
