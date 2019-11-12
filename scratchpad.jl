

spl_training = view(training, sample(axes(training, 1), 100; replace = false, ordered = true), :)

spl_P = P[spl_training[:,:userN], :]
spl_Q = Q[spl_training[:,:movieN], :]


spl_P = P[spl_training[:,:userN], :]
spl_Q = Q[spl_training[:,:movieN], :]

# Gradient
(ΔP, ΔQ) = gradient((p, q) -> J(p, q), P, Q)[spl_P, spl_Q]

P[spl_training[:,:userN], :]  = spl_P - α * ΔP
Q[spl_training[:,:movieN], :] = spl_Q - α * ΔQ


P = zeros(Float64, 2, 1)
Q = zeros(Float64, 2, 1)

function J(matrix1::Array{Float64, 2}, matrix2::Array{Float64, 2})
    return sum((([1; 1] - P' * Q).^2) / 2, 2)
end

gradient((p, q) -> J(p, q), P, Q)[1; 1]
