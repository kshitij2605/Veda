# Test fixtures

Files here are loaded via `include_str!` from unit tests only. They are
never read at runtime.

- `jwt_test_priv.pem` / `jwt_test_pub.pem` — a 2048-bit RSA keypair used to
  sign JWTs in `auth_oidc::tests`. Regenerated with `openssl genrsa` on
  every machine that needs to update them; the values themselves are not
  privileged because they never authenticate against anything real.
