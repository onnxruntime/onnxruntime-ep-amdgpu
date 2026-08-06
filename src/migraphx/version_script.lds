#_init and _fini should be local
VERS_1.0 {
  global:
    CreateEpFactories;
    ReleaseEpFactory;
    GetBackendTelemetry;
  # Hide everything else.
  local:
    *;
};
