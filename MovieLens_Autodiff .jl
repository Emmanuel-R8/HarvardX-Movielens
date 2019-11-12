# using ZipFile,
using CSV
using DataFrames
using StatsBase
using Random, Distributions
using Flux

# Import ratings file
cd(raw"/home/emmanuel/Development/Learning/capstone-movielens/")
# z = ZipFile.Reader("datasets/ratings.zip")
# ratings = CSV.read(z.files[1];
#                 delim="::",
#                 header=["userId", "movieId", "rating", "timestamp"],
#                 types = [Int, Int, Float64, Int])

edx = CSV.read("datasets/edx.csv";
               delim=",",
               copycols=true,
               datarow=2,
               header=["rowN", "userId", "movieId", "rating", "timestamp", "title", "genres"],
               types = [Int, Int, Int, Float64, Int, String, String])

validation = CSV.read("datasets/validation.csv";
                      delim=",",
                      copycols=true,
                      datarow=2,
                      header=["rowN", "userId", "movieId", "rating", "timestamp", "title", "genres"],
                      types = [Int, Int, Int, Float64, Int, String, String])

r_m, r_sd = mean_and_std(edx.rating)

edx.rating_z        = (edx.rating .- r_m) ./ r_sd
validation.rating_z = (validation.rating .- r_m) ./ r_sd

movieIndex = DataFrame(movieId = unique(edx, :movieId).movieId)
nMovie = size(movieIndex)[1]
movieIndex.movieN = 1:nMovie

userIndex = DataFrame(userId = unique(edx, :userId).userId)
nUser = size(userIndex)[1]
userIndex.userN = 1:nUser

movieMean = by(edx, :movieId, movie_mean = :rating_z => mean)
userMean = by(edx, :userId, user_mean = :rating_z => mean)

training = join(edx, movieIndex, on = :movieId )
training = join(training, userIndex, on = :userId )
training = training[:, [:userN, :movieN, :rating_z]]
nSamples = size(training)[1]

test = join(validation, movieIndex, on = :movieId )
test = join(test, userIndex, on = :userId )
test = test[:, [:userN, :movieN, :rating, :rating_z]]
nTest = size(test)[1]

nLF = 3

P = zeros(Float64, nUser, nLF)
Q = zeros(Float64, nMovie, nLF)

P[:,1] .= 1
P[:,2] = sort(join(userIndex, userMean, on = :userId, kind = :left), :userN)[:, :user_mean]

Q[:,1] = sort(join(movieIndex, movieMean, on = :movieId, kind = :left), :movieN)[:, :movie_mean]
Q[:,2] .= 1

Random.seed!(42)
P[:, 3] = rand(Normal(), nUser) / 1000
Q[:, 3] = rand(Normal(), nMovie) / 1000


# Make P and Q Flux parameters
P = param(P)
Q = param(Q)


# Cost function on sample

function J(p, q)
    error = sum( p[test[:,:userN], :] .* q[test[:,:movieN], :]; dims = 2) .* r_sd .+ r_m
    error = sqrt( sum((test.rating - error).^2) / nTest )
    
    return error
end

P[13, :]

# Cost function on full set

function prediction(user_n, movie_n) = sum( P[user_n, :] .* Q[movie_n, :], dims = 2) .* r_sd + r_m 
    

function J(p, q)
    error = sum( p[training[:,:userN], :] .* q[training[:,:movieN], :]; dims = 2) .* r_sd .+ r_m
    error = sqrt( sum((training.rating - error).^2) / nTest )
    
    return error
end
 

# Gradient

dJ = Tracker.gradient(() -> J(p, q), Flux.params(P, Q), nest = true)




function stochastic_grad_descent(P::Array{Float64, 2}, Q::Array{Float64, 2}, grad;
  times = 1, batch_size = 10000, λ = 0.1, α = 0.01, verbose = true)

  for i = 1:times
        
    spl_P = P[spl_training[:,:userN], :]
    spl_Q = Q[spl_training[:,:movieN], :]
    
       
        
    P[spl_training[:,:userN], :]  = spl_P - α * ΔP
    Q[spl_training[:,:movieN], :] = spl_Q - α * ΔQ
  end # for loop

  return P, Q
end

float_t = sum( P[test[:,:userN], :] .* Q[test[:,:movieN], :]; dims = 2) .* r_sd .+ r_m
float_t = sqrt( sum((test.rating - float_t).^2) / nTest )
println("Initial floating point RMSE test = ",float_t)

round_t = round.(P[test[:,:userN], :] .* Q[test[:,:movieN], :])
round_t = sum(round_t ; dims = 2) .* r_sd .+ r_m
round_t = sqrt( sum((test.rating - round_t).^2) / nTest )
println("Initial rounded RMSE test = ", round_t)

starting_α = 0.01
validation_results =[0 float_t round_t]

for n = 1:500
  batch_size = 10000
  α = starting_α
  nFeatures = size(P)[2]
  λ = 0.1 * (nUser + nMovie) * nFeatures / 2_000_000

  RMSE = sqrt(sum((training.rating_z -
                  sum( P[training[:,:userN], :] .* Q[training[:,:movieN], :]; dims = 2)).^2) / nSamples)

  for i = 1:250
    RMSE_tmp =   RMSE

    (newP, newQ) = stochastic_grad_descent(P, Q; times = 100, batch_size = 1000 * nFeatures, λ = λ, α = α)
    global P = newP
    global Q = newQ

    RMSE = sqrt(sum((training.rating_z -
                     sum( P[training[:,:userN], :] .* Q[training[:,:movieN], :]; dims = 2)).^2) / nSamples)

    println("Step: ", i, "   RMSE of z-score training = ", RMSE)

    if (RMSE > RMSE_tmp)
      α /= 2
      println("α decreased to ", α)
    end

    if (starting_α / α > 10000) | (abs((RMSE - RMSE_tmp) / RMSE_tmp) < 1e-6)
      break
    end

  end


  test_RMSE = P[test[:,:userN], :] .* Q[test[:,:movieN], :]

  # floating point ratings float_RMSE = sum(test_RMSE ; dims = 2).* r_sd .+ r_m float_RMSE = sqrt( sum((test.rating - float_RMSE).^2) / nTest ) println("Step: ", n, "    float RMSE test = ",float_RMSE, " with number features = ", nFeatures) # Round to only obtain legal ratings round_RMSE = sum( round.(test_RMSE) ; dims = 2).* r_sd .+ r_m round_RMSE = sqrt( sum((test.rating - round_RMSE).^2) / nTest ) println("Step: ", n, "    round RMSE test = ",round_RMSE, " with number features = ", nFeatures) global validation_results = [validation_results; nFeatures float_RMSE round_RMSE] # Add 1 features global P = [P rand(Normal(), nUser)/1000] global Q = [Q rand(Normal(), nMovie)/1000]
  nFeatures += 1

end

function stochastic_grad_descent_simple(P::Array{Float64, 2}, Q::Array{Float64, 2};
  times = 1, batch_size = 10000, λ = 0.1, α = 0.01, verbose = true)

  for i = 1:times
    spl_training = view(training, sample(axes(training, 1), batch_size; replace = false, ordered = true), :)

    spl_P = P[spl_training[:,:userN], :]
    spl_Q = Q[spl_training[:,:movieN], :]

    err = spl_training.rating_z - sum(spl_P .* spl_Q; dims = 2)
    ΔP = - 2 * err .* spl_Q + λ * spl_P
    ΔQ = - 2 * err .* spl_P + λ * spl_Q

    P[spl_training[:,:userN], :]  = spl_P - α * ΔP
    Q[spl_training[:,:movieN], :] = spl_Q - α * ΔQ
  end # for loop

  return P, Q
end
