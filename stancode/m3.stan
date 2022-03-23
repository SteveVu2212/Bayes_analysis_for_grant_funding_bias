data{
  array[18] int<lower=0> A;
  array[18] int<lower=0> N;
  array[18] int<lower=1,upper=2> G;
  array[18] int<lower=1,upper=9> D;
}

parameters{
  matrix[2,9] z;
  real a_bar;
  real<lower=0> sigma;
}

transformed parameters{
  array[18] real<lower=0,upper=1> p;
  for(i in 1:18){
    p[i] = a_bar + z[G[i],D[i]]*sigma;
    p[i] = inv_logit(p[i]);
  }
}

model{
  a_bar ~ normal(0,1.5);
  sigma ~ exponential(1.5);
  to_vector(z) ~ normal(0, 1);
  A ~ binomial(N, p);
}

generated quantities{
  matrix[2,9] a;
  vector[18] log_lik;
  for(i in 1:2){
    for(j in 1:9){
      a[i,j] = a_bar + z[i,j]*sigma;
    }
  }
  for(i in 1:18){
    log_lik[i] = binomial_lpmf(A[i] | N[i], p[i] );
  }
}









