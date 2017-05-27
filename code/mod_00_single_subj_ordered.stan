data {
  int N;           # Total number of observations
  vector[N] TIME;  # TIME for each data
  real DOSE;       # DOSE for each subject
  vector[N] Y;     # Observation
}

parameters {
  #real<lower=0> KA;  # Absorption rate constant (ka)
  #real<lower=0> CL;  # Clearance (CL)
  real<lower=0> VD;  # Volume of distribution (Vd)
  real<lower=0> s_Y; # SD of Y in log scale
  positive_ordered[2] K;
}

transformed parameters {
  real KA;      # Absorption rate constant (ka)
  real KEL;     # Elimination rate constant (kel)
  vector[N] mu; # Calculated concentration
  
  KA  = K[2];
  KEL = K[1];
  
  # Analytical solution of 1 compartment model
  mu = DOSE / VD * KA * (exp(-KA * TIME)-exp(-KEL * TIME))/(KEL-KA);
}

model {
  K[1] ~ lognormal(log(0.1), 1);
  K[2] ~ lognormal(log(0.3), 1);
  
  # Assume Y follows log-normal distribution
  Y  ~ lognormal(log(mu),  s_Y);
}

generated quantities {
  vector[N] y_new;
  real CL;  # Clearance (CL)
  
  # Calculate CL from kel and Vd
  CL = KEL * VD;
  
  for (n in 1:N){
    y_new[n]  = lognormal_rng(log(mu[n]), s_Y);
  }
}