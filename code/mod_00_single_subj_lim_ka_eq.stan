data {
  int N;           // Total number of observations
  vector[N] TIME;  // TIME for each data
  real DOSE;       // DOSE for each subject
  vector[N] Y;     // Observation
}

parameters {
  real<lower=0> CL;  // Clearance (CL)
  real<lower=0> VD;  // Volume of distribution (Vd)
  real<lower=CL/VD> KA;  // Absorption rate constant (ka)
  real<lower=0> s_Y; // SD of Y in log scale
}

transformed parameters {
  real KEL;     // Elimination rate constant (kel)
  vector[N] mu; // Calculated concentration
  
  // Calculate kel from CL and Vd
  KEL = CL / VD;
  
  // Analytical solution of 1 compartment model
  mu = DOSE / VD * KA * (exp(-KA * TIME)-exp(-KEL * TIME))/(KEL-KA);
}

model {
  // Weak priors
  KA ~ lognormal(log(0.5), 1);
  CL ~ lognormal(log(0.5), 1);
  VD ~ lognormal(log(5),   1);
  
  // Assume Y follows log-normal distribution
  Y  ~ lognormal(log(mu),  s_Y);
}

generated quantities {
  vector[N] y_new;
  
  for (n in 1:N){
    y_new[n]  = lognormal_rng(log(mu[n]), s_Y);
  }
}
