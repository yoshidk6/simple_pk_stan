data {
  # For data
  int N;    # Total number of observations
  vector[N] TIME;  # TIME for each data
  real DOSE; # DOSE for each subject
  vector[N] Y;     # Observation
  
  real KA;
  
  # For Prediction
  int N_new;
  vector[N_new] TIME_new;
}

parameters {
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
  Y ~ lognormal(log(mu), s_Y);
}

generated quantities {
  vector[N_new] mu_new;
  vector[N_new] y_new;
  
  mu_new = DOSE / VD * KA * (exp(-KA * TIME_new)-exp(-KEL * TIME_new))/(KEL-KA);
  
  for (n in 1:N_new){
    y_new[n]  = lognormal_rng(log(mu_new[n]), s_Y);
  }
}
