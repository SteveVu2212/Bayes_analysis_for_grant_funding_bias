data{
  array[18] int<lower=0> A;
  array[18] int<lower=0> N;
  array[18] int<lower=1,upper=2> G;
  array[18] int<lower=1,upper=9> D;
}

parameters{
  matrix[2,9] a;
}

transformed parameters{
  array[18] real<lower=0,upper=1> p;
  for(i in 1:18){
    p[i] = a[G[i],D[i]];
    p[i] = inv_logit(p[i]);
  }
}

model{
  to_vector(a) ~ normal(0,1.5);
  A ~ binomial(N, p);
}

generated quantities{
  vector[18] log_lik;
  for(i in 1:18){
    log_lik[i] = binomial_lpmf(A[i] | N[i], p[i] );
  }
}