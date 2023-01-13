struct Squared{N} <: Bijector{N} end
(::Squared)(x) = x.^2 # transform itself, "forward"
(::Inverse{<: Squared})(y) = y.^0.5 # inverse tramsform, "backward"
logabsdetjac(::Squared{0}, y::Real) = log(convert(eltype(y),2)) + log(y) # ∂ₓx^2 = 2 x → log(abs(1)) = log(2) + log(x) # if we do not want 

struct Abs{N} <: Bijector{N} end
(::Abs)(x) = abs.(x) # transform itself, "forward"
(::Inverse{<: Abs})(y) = y # inverse tramsform, "backward"
logabsdetjac(::Abs{0}, y::Real) = zero(eltype(y)) # ∂ₓx^2 = 2 x → log(abs(1)) = log(2) + log(x)

struct NegAbs{N} <: Bijector{N} end
(::NegAbs)(x) = - abs.(x) # transform itself, "forward"
(::Inverse{<: NegAbs})(y) = y # inverse tramsform, "backward"
logabsdetjac(::NegAbs{0}, y::Real) = zero(eltype(y)) # ∂ₓx^2 = 2 x → log(abs(1)) = log(2) + log(x)