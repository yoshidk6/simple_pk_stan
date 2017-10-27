data {
  // For data
  int N;    // Total number of observations
  int N_ID; // Total number of subjects
  int ID[N];    // ID for each data
  vector[N] TIME;  // TIME for each data
  vector[N_ID] DOSE; // DOSE for each subject
  vector[N_ID] WT; // WT for each subject
  real WTMED; // Median WT
  vector[N] Y;     // Observation
}

parameters {
  // Typical parameters
  real<lower=0> KA;
  real<lower=0> CL;
  real<lower=0> VD;
  
  // IIV and residual error
  corr_matrix[2] rho;
  vector<lower=0>[2] sigma;
  real<lower=0> s_Y;
  
  // Covariate
  real WTCL;
  real WTVD;
  
  // Individual parameters
  vector[2] CLVDiLog[N_ID];
}

transformed parameters {
  vector<lower=0>[N] mu;
  
  // Transform to each time point
  vector<lower=0>[N] DOSE_N;
  vector<lower=0>[N] CL_N;
  vector<lower=0>[N] VD_N;
  vector<lower=0>[N] KEL_N;
  
  // Individual parameters
  vector<lower=0>[N_ID] CLi;
  vector<lower=0>[N_ID] VDi;
  
  // Inputs for multi_normal
  vector[2] CLVD;
  cov_matrix[2] Omega;
  
  // Send to multi_normal in model block
  CLVD[1] = CL;
  CLVD[2] = VD;
  
  Omega = quad_form_diag(rho, sigma);
  
  // Translate estimates to each parameter vectors
  // and add covariate effect
  for (k in 1:N_ID){
    CLi[k] = exp(CLVDiLog[k,1]) * (WT[k]/WTMED)^WTCL;
    VDi[k] = exp(CLVDiLog[k,2]) * (WT[k]/WTMED)^WTVD;
  }
  
  // Assign individual parameters for each time point
  
  //// Extract individual PK parameters

  DOSE_N = DOSE[ID];
  CL_N   = CLi[ID];
  VD_N   = VDi[ID];
  
  KEL_N  = CL_N ./ VD_N;
  
  // Solver ODE
  mu = DOSE_N ./ VD_N * KA .* (exp(-KA * TIME)-exp(-KEL_N .* TIME)) ./ (KEL_N-KA);
  
}

model {
  KA ~ lognormal(log(0.5), 1);
  CL ~ lognormal(log(0.5), 1);
  VD ~ lognormal(log(5),   1);
  
  sigma[1] ~ cauchy(0, 1); // IIV on CL
  sigma[2] ~ cauchy(0, 1); // IIV on VD
  rho ~ lkj_corr(1); 
  
  s_Y  ~ cauchy(0, 1);
  
  WTCL  ~ normal(0, 1);
  WTVD  ~ normal(0, 1);
  
  CLVDiLog ~ multi_normal(log(CLVD), Omega);
  Y   ~ lognormal(log(mu), s_Y);
}

generated quantities {
  vector[N] mu_new;
  vector[N] y_new;
  real cor;

  vector[N] mu_newPred;
  vector[N] y_newPred;
  vector[2] CLVDiPredLog[N_ID];
  vector<lower=0>[N_ID] CLiPred;
  vector<lower=0>[N_ID] VDiPred;
  vector<lower=0>[N] CL_NPred;
  vector<lower=0>[N] VD_NPred;
  vector<lower=0>[N] KEL_NPred;

  // Calculate correlation coefficient
  cor = Omega[2,1] / (sigma[1] * sigma[2]);


  // Prediction with individual parameter estimates
  //// For calculating prediction intervals
  mu_new = DOSE_N ./ VD_N * KA .* (exp(-KA * TIME)-exp(-KEL_N .* TIME)) ./ (KEL_N-KA);
  for (n in 1:N){
    y_new[n]  = lognormal_rng(log(mu_new[n]), s_Y);
  }


  // Prediction with population parameter estimates + IIV
  for(k in 1:N_ID){
    CLVDiPredLog[k] = multi_normal_rng(log(CLVD), Omega);
    CLiPred[k] = exp(CLVDiPredLog[k,1]) * (WT[k]/WTMED)^WTCL;
    VDiPred[k] = exp(CLVDiPredLog[k,2]) * (WT[k]/WTMED)^WTVD;
  }

  CL_NPred   = CLiPred[ID];
  VD_NPred   = VDiPred[ID];
  KEL_NPred  = CL_NPred ./ VD_NPred;

  mu_newPred = DOSE_N ./ VD_NPred * KA .* (exp(-KA * TIME)-exp(-KEL_NPred .* TIME)) ./ (KEL_NPred-KA);
  for (n in 1:N){
    y_newPred[n]  = lognormal_rng(log(mu_newPred[n]), s_Y);
  }
}
