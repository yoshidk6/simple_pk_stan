data {
  // For data
  int N;    // Total number of observations
  int N_ID; // Total number of subjects
  int ID[N];    // ID for each data
  vector[N] TIME;  // TIME for each data
  vector[N_ID] DOSE; // DOSE for each subject
  vector[N] Y;     // Observation
}

parameters {
  real<lower=0> KA;
  real<lower=0> CL;
  real<lower=0> VD;
  
  vector<lower=0>[N_ID] CLi;
  vector<lower=0>[N_ID] VDi;
  
  real<lower=0> s_CL;
  real<lower=0> s_VD;
  real<lower=0> s_Y;

}

transformed parameters {
  vector<lower=0>[N] mu;
  vector<lower=0>[N] DOSE_N;
  vector<lower=0>[N] CL_N;
  vector<lower=0>[N] VD_N;
  vector<lower=0>[N] KEL_N;
  
  DOSE_N = DOSE[ID];
  CL_N   = CLi[ID];
  VD_N   = VDi[ID];
  
  KEL_N  = CL_N ./ VD_N;
  
  mu = DOSE_N ./ VD_N * KA .* (exp(-KA * TIME)-exp(-KEL_N .* TIME)) ./ (KEL_N-KA);
  
}

model {
  // Weak priors
  KA ~ lognormal(log(0.5), 1);
  CL ~ lognormal(log(0.5), 1);
  VD ~ lognormal(log(5),   1);
  
  // Assume CLi, VDi, and Y follows log-normal distribution
  CLi ~ lognormal(log(CL), s_CL);
  VDi ~ lognormal(log(VD), s_VD);
  Y   ~ lognormal(log(mu), s_Y);
}

generated quantities {
  vector[N] mu_new;
  vector[N] y_new;
  
  vector[N] mu_newPred;
  vector[N] y_newPred;
  vector<lower=0>[N_ID] CLiPred;
  vector<lower=0>[N_ID] VDiPred;
  vector<lower=0>[N] CL_NPred;
  vector<lower=0>[N] VD_NPred;
  vector<lower=0>[N] KEL_NPred;
  
  // Prediction with individual parameter estimates
  mu_new = DOSE_N ./ VD_N * KA .* (exp(-KA * TIME)-exp(-KEL_N .* TIME)) ./ (KEL_N-KA);
  for (n in 1:N){
    y_new[n]  = lognormal_rng(log(mu_new[n]), s_Y);
  }
  
  
  // Prediction with population parameter estimates + IIV
  for(n_id in 1:N_ID){
    CLiPred[n_id] = lognormal_rng(log(CL), s_CL);
    VDiPred[n_id] = lognormal_rng(log(VD), s_VD);
  }
  CL_NPred   = CLiPred[ID];
  VD_NPred   = VDiPred[ID];
  KEL_NPred  = CL_NPred ./ VD_NPred;
  
  mu_newPred = DOSE_N ./ VD_NPred * KA .* (exp(-KA * TIME)-exp(-KEL_NPred .* TIME)) ./ (KEL_NPred-KA);
  for (n in 1:N){
    y_newPred[n]  = lognormal_rng(log(mu_newPred[n]), s_Y);
  }
    
}
