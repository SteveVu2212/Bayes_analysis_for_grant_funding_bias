data{
  int D[18];
  int G[18];
  int A[18];
  int N[18];

}
parameters{
  corr_matrix[2] Rho;
  vector<lower=0>[2] sigma;
  real a_bar;
  real b_bar;
  vector[9] a;
  vector[9] b;
}
model{
  vector[18] p;
  sigma ~ normal(0,1);
  Rho ~ lkj_corr(2);
  a_bar ~ normal(0,1.5);
  b_bar ~ normal(0,1.5);
  {
  vector[2] MU;
  vector[2] YY[9];
  
  MU = [a_bar, b_bar]';
  for(i in 1:9){YY[i] = [a[i], b[i]]';}
  YY ~ multi_normal(MU, quad_form_diag(Rho, sigma));
  }
  for(i in 1:18){
    p[i] = a[D[i]] + b[D[i]]*G[i];
    p[i] = inv_logit(p[i]);
  }
  A ~ binomial(N, p);
}
// generated quantities{
//   vector[N] log_lik;
//   vector[N] p;
//   for(i in 1:N){
//     p[i] = a[dept[i]] + b[dept[i]]*male[i];
//     p[i] = inv_logit(p[i]);
//   }
//   for(i in 1:N){
//     log_lik[i] = binomial_lpmf(admit[i] | applications[i], p[i]);
//   }
// }