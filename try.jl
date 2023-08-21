# using LinearAlgebra

# struct MultiCopterState
#     position::Vector{Float64}
#     velocity::Vector{Float64}
#     attitude::Vector{Float64}
#     angular_velocity::Vector{Float64}
# end

# function dynamics(state::MultiCopterState, controls::Vector{Float64}, dt::Float64)
#     # Constants
#     mass = 1.0
#     gravity = [0.0, 0.0, -9.81]

#     # Unpack state variables
#     position, velocity, attitude, angular_velocity = state

#     # Unpack control inputs
#     motor_speeds = controls

#     # Calculate forces and torques
#     thrust = sum(motor_speeds)
#     forces = [0.0, 0.0, thrust] - mass * gravity
#     torques = cross([0.0, 0.0, sum(motor_speeds[1:2]) - motor_speeds[3]], [0.0, 0.0, 1.0])

#     # Update position and velocity
#     acceleration = forces / mass
#     position += velocity * dt + 0.5 * acceleration * dt^2
#     velocity += acceleration * dt

#     # Update attitude and angular velocity
#     angular_acceleration = torques / mass
#     attitude += angular_velocity * dt + 0.5 * angular_acceleration * dt^2
#     angular_velocity += angular_acceleration * dt

#     # Create and return updated state
#     return MultiCopterState(position, velocity, attitude, angular_velocity)
# end

# function run_simulation()
#     controls = [10.0, 10.0, 10.0]
#     dt = 0.01

#     initial_state = MultiCopterState([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
#     # Simulation loop
#     for i in 1:100
#         initial_state = dynamics(initial_state, controls, dt)
#         println("Position: $(initial_state.position), Velocity: $(initial_state.velocity)")
#     end
# end

# run_simulation()

# import Pkg; 
# Pkg.add("DifferentialEquations")
# Pkg.add("Plots")

using DifferentialEquations
using Plots

# Define the drone's state
mutable struct Drone
    position::Array{Float64,1}
    velocity::Array{Float64,1}
    orientation::Array{Float64,1} # roll, pitch, yaw in radians
    angular_velocity::Array{Float64,1}
end

# Initialize a drone
drone = Drone([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

# Define the drone's dynamics (e.g., movement equations)
function drone_dynamics!(du, u, p, t)
    # Placeholder: In a realistic model, this function would consider many factors, including current velocity,
    # orientation, aerodynamics, control inputs, etc.
    
    gravity = 9.81 # gravitational constant, m/s^2
    mass = 1.0 # drone mass in kg

    du[1:3] = u[4:6] # position updates with velocity
    du[4:6] = [0.0, 0.0, -gravity] + p/mass # velocity updates with acceleration due to forces
    du[7:9] = u[10:12] # orientation updates with angular velocity
    du[10:12] = [0.0, 0.0, 0.0] # placeholder for angular velocity change, consider moments of inertia and torques
end

# Define the initial conditions for the simulation
forces = [0.0, 0.0, 10.0] # forces in x, y, z directions in N
u0 = [drone.position; drone.velocity; drone.orientation; drone.angular_velocity]
tspan = (0.0, 10.0)  # Simulate for 10 seconds

# Define the problem and solve it
problem = ODEProblem(drone_dynamics!, u0, tspan, forces)
solution = solve(problem)

# The solution object contains the state of the drone at each time step.
# Access this data to analyze the drone's behavior over time.
# println(solution)

# Plot the drone's position over time
plt = plot3d(
    1,
    vars=(1,2,3),
    xlabel="x (m)",
    ylabel="y (m)",
    zlabel="z (m)",
    title="Drone Position Over Time"
)

@time plot(plt)

@gif for i in 1:100
    plot(plt)
end every 1
# # # Plot the drone's velocity over time
# plt = plot3d(
#     solution,
#     vars=(4,5,6),
#     xlabel="x (m/s)",
#     ylabel="y (m/s)",
#     zlabel="z (m/s)",
#     title="Drone Velocity Over Time"
# )

# @time plot(plt)
# @gif for i in 1:100
#     plot(plt)
# end
