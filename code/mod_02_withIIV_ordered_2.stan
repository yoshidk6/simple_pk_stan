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
  real<lower=0> VD;
  real<lower=0> KEL;
  real<lower=0> KA_KEL_log;
  
  real<lower=0> s_CL;
  real<lower=0> s_VD;
  real<lower=0> s_Y;
  
  vector<lower=0>[N_ID] CLi;
  vector<lower=0>[N_ID] VDi;

}

transformed parameters {
  real<lower=0> KA;
  real<lower=0> CL;
  vector<lower=0>[N] mu;
  vector<lower=0>[N] DOSE_N;
  vector<lower=0>[N] CL_N;
  vector<lower=0>[N] VD_N;
  vector<lower=0>[N] KEL_N;
  
  # Calc KA and CL
  KA = KEL * exp(KA_KEL_log);
  CL = KEL * VD;
  
  # Assign individual parameters to each data
  DOSE_N = DOSE[ID];
  CL_N   = CLi[ID];
  VD_N   = VDi[ID];
  
  KEL_N  = CL_N ./ VD_N;
  
  mu = DOSE_N ./ VD_N * KA .* (exp(-KA * TIME)-exp(-KEL_N .* TIME)) ./ (KEL_N-KA);
  
}

model {
  KA_KEL_log ~ lognormal(log(log(10)), 1);
  KEL ~ lognormal(log(0.05), 1);
  #VD ~ lognormal(log(6), 1);
  #s_CL ~ lognormal(log(0.2), 1);
  #s_VD ~ lognormal(log(0.2), 1);
  #s_Y  ~ lognormal(log(0.2), 1);
  
  CLi ~ lognormal(log(CL), s_CL);
  VDi ~ lognormal(log(VD), s_VD);
  Y   ~ lognormal(log(mu), s_Y);
}

generated quantities {
  vector[N] mu_new;
  vector[N] y_new;
  
  mu_new = DOSE_N ./ VD_N * KA .* (exp(-KA * TIME)-exp(-KEL_N .* TIME)) ./ (KEL_N-KA);
  
  for (n in 1:N){
    y_new[n]  = lognormal_rng(log(mu_new[n]), s_Y);
  }
}
