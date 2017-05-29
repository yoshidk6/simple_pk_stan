data {
  # For data
  int N;    # Total number of observations
  int N_ID; # Total number of subjects
  int ID[N];    # ID for each data
  vector[N] TIME;  # TIME for each data
  vector[N_ID] DOSE; # DOSE for each subject
  vector[N] Y;     # Observation
  real VD_low;     # Lower limit of VD
}

parameters {
  ## Typical parameters
  real<lower=0> KA;
  real<lower=0> CL;
  real<lower=VD_low> VD;
  
  ## Individual parameters
  vector<lower=0>[2] CLVDi_log[N_ID];
  
  ## Deviations and a covariance
  real<lower=0> s_CL;
  real<lower=0> s_VD;
  real cov_CLVD;      # Covariance between log(CL) and log(VD)
  real<lower=0> s_Y;
}

transformed parameters {
  vector<lower=0>[N] mu;
  
  ## Transform to each time point
  vector<lower=0>[N] DOSE_N;
  vector<lower=0>[N] CL_N;
  vector<lower=0>[N] VD_N;
  vector<lower=0>[N] KEL_N;
  
  ## Inputs for multi_normal
  vector[2] CLVD;
  cov_matrix[2] sigma;
  
  ## Individual parameters
  vector<lower=0>[N_ID] CLi;
  vector<lower=VD_low>[N_ID] VDi;
  
  ## Send to multi_normal in model block
  CLVD[1] = CL;
  CLVD[2] = VD;
  
  sigma[1,1] = s_CL^2;
  sigma[2,1] = cov_CLVD;
  sigma[1,2] = cov_CLVD;
  sigma[2,2] = s_VD^2;
  
  ## Extract individual PK parameters
  ## Not sure if indexing in N_ID dimension is allowed
  ## Couldn't set lower limit for VDi in CLVDi_log, so manually adding it
  for (k in 1:N_ID){
    CLi[k] = exp(CLVDi_log[k,1]);
    VDi[k] = exp(CLVDi_log[k,2]);
  }
  
  ## Assign individual parameters for each time point
  DOSE_N = DOSE[ID];
  CL_N   = CLi[ID];
  VD_N   = VDi[ID];
  
  KEL_N  = CL_N ./ VD_N;
  
  mu = DOSE_N ./ VD_N * KA .* (exp(-KA * TIME)-exp(-KEL_N .* TIME)) ./ (KEL_N-KA);
  
}

model {
  KA ~ lognormal(log(0.5), 1);
  CL ~ lognormal(log(0.5), 1);
  VD ~ lognormal(log(5),   1);
  
  CLVDi_log ~ multi_normal(log(CLVD), sigma);
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
