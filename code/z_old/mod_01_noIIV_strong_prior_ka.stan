data {
  # For data
  int N;    # Total number of observations
  int N_ID; # Total number of subjects
  int ID[N];    # ID for each data
  vector[N] TIME;  # TIME for each data
  vector[N_ID] DOSE; # DOSE for each subject
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
  vector[N] DOSE_N;
  real KEL;
  
  KEL= CL/VD;
  DOSE_N = DOSE[ID];
  
  mu = DOSE_N / VD * KA .* (exp(-KA * TIME)-exp(-KEL * TIME))/(KEL-KA);
  
}

model {
  KA ~ normal(1, 0.1);
  Y ~ lognormal(log(mu), s_Y);
}

generated quantities {
  vector[N] mu_new;
  vector[N] y_new;
  
  mu_new = DOSE_N / VD * KA .* (exp(-KA * TIME)-exp(-KEL * TIME))/(KEL-KA);
  
  for (n in 1:N){
    y_new[n]  = lognormal_rng(log(mu_new[n]), s_Y);
  }
}
