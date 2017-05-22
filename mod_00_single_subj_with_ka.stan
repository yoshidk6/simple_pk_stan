data {
  int N;           # Total number of observations
  vector[N] TIME;  # TIME for each data
  real DOSE;       # DOSE for each subject
  vector[N] Y;     # Observation
}


parameters {
  real<lower=0> KA;
  real<lower=0> CL;
  real<lower=0> VD;
  real<lower=0> s_Y;
}


transformed parameters {
  vector[N] mu;
  real KEL;
  
  KEL = CL / VD;
  
  mu = DOSE / VD * KA * (exp(-KA * TIME)-exp(-KEL * TIME))/(KEL-KA);
}


model {
  KA ~ lognormal(log(0.3), 0.1);
  Y  ~ lognormal(log(mu),  s_Y);
}


generated quantities {
  vector[N] y_new;
  
  for (n in 1:N){
    y_new[n]  = lognormal_rng(log(mu[n]), s_Y);
  }
}
