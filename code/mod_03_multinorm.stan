data {
  # For data
  int N;    # Total number of observations
  int N_ID; # Total number of subjects
  int ID[N];    # ID for each data
  vector[N] TIME;  # TIME for each data
  vector[N_ID] DOSE; # DOSE for each subject
  vector[N] Y;     # Observation
  real KA;     # Fix KA
}

parameters {
  # Typical parameters
  real<lower=0> CL;
  real<lower=0> VD;
  
  # Individual parameters
  vector<lower=0>[2] CLVDi[N_ID];
  
  # Deviations and a covariance
  real<lower=0> s_CL;
  real<lower=0> s_VD;
  real cov_CLVD;      # Covariance between log(CL) and log(VD)
  real<lower=0> s_Y;
}

transformed parameters {
  vector<lower=0>[N] mu;
  
  # Transform to each time point
  vector<lower=0>[N] DOSE_N;
  vector<lower=0>[N] CL_N;
  vector<lower=0>[N] VD_N;
  vector<lower=0>[N] KEL_N;
  
  # Individual parameters
  vector<lower=0>[N_ID] CLi;
  vector<lower=0>[N_ID] VDi;
  
  # Inputs for multi_normal
  vector[2] CLVD;
  cov_matrix[2] sigma;
  
  # Send to multi_normal in model block
  CLVD[1] = CL;
  CLVD[2] = VD;
  
  sigma[1,1] = s_CL^2;
  sigma[2,1] = cov_CLVD;
  sigma[1,2] = cov_CLVD;
  sigma[2,2] = s_VD^2;
  
  # Translate estimates to each parameter vectors
  for (k in 1:N_ID){
    CLi[k] = CLVDi[k,1];
    VDi[k] = CLVDi[k,2];
  }
  
  # Assign individual parameters for each time point
  
  ## Extract individual PK parameters
  ### Not sure if indexing in row dimension is allowed
  ### Couldn't set lower limit for VDi in CLVDi_log, so manually limiting it for CL_N and VD_N
  
  DOSE_N = DOSE[ID];
  CL_N   = CLi[ID];
  VD_N   = VDi[ID];
  
  KEL_N  = CL_N ./ VD_N;
  
  # Solver ODE
  mu = DOSE_N ./ VD_N * KA .* (exp(-KA * TIME)-exp(-KEL_N .* TIME)) ./ (KEL_N-KA);
  
}

model {
  CL ~ lognormal(log(0.5), 1);
  VD ~ lognormal(log(5),   1);
  
  s_CL ~ normal(0, 1);
  s_VD ~ normal(0, 1);
  cov_CLVD ~ normal(0, 1);
  
  CLVDi ~ multi_normal_log(log(CLVD), sigma);
  Y   ~ lognormal(log(mu), s_Y);
}

generated quantities {
  vector[N] mu_new;
  vector[N] y_new;
  real cor;
  
  ## For calculating prediction intervals
  mu_new = DOSE_N ./ VD_N * KA .* (exp(-KA * TIME)-exp(-KEL_N .* TIME)) ./ (KEL_N-KA);
  for (n in 1:N){
    y_new[n]  = lognormal_rng(log(mu_new[n]), s_Y);
  }
  
  ## Calculate correlation coefficient
  cor = cov_CLVD / (s_CL * s_VD);
}
