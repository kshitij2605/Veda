# A node whose `raft_state.bin` fails to load becomes a silent non-voter

**2026-09-09.** Observed while bringing up the 3-node cluster for the first
24-hour soak. Not a live defect on any normal path, but the failure mode is
bad enough to write down: the node stays **healthy on every check while
participating in nothing.**

## What happened

The cluster was started from a freshly built image against **pre-existing
volumes** written months earlier by an older binary. Node-1's startup:

```
INFO  raft WAL replayed  wal_records=2310 entries=2308
      last_log_id=Some(LogId { leader_id: { term: 1411, node_id: 2 },
                               index: 2618142 })
WARN  raft state deserialise failed, starting empty
      error="io error: "  path="/var/veda/data/raft/raft_state.bin"
INFO  get_initial_state  vote=T1411-N2:committed  last_applied=None
INFO  startup done: id=1 target_state: Learner
      membership_state: { configs: [], nodes: {} }
      is_leader: false, is_voter: false
```

`GET /v1/admin/nodes` then returned `{"mode":"raft","nodes":[]}` on all three,
and no leader was ever elected.

## Why it wedges

Three behaviours compose badly:

1. **A failed state-file load is non-fatal.** `raft_state.bin` fails to
   deserialise, the node logs a WARN and continues with empty vote and
   membership.
2. **Bootstrap is skipped when the log is non-empty.** `VEDA_RAFT_BOOTSTRAP=1`
   only initialises a node that has nothing — correct in general, since
   re-initialising a node with history would be far worse. Here the WAL
   replayed 2.6M entries, so bootstrap declined.
3. **The result is a voter-less Learner that nothing recovers.** It has a log
   but no membership, so it cannot vote, cannot be voted for, and has no peers
   to learn membership from — every other node is in the same state.

**And `/v1/admin/health` returns 200 throughout.** The process is up, the HTTP
surface works, the storage layer is fine. Only `/admin/nodes` shows the
cluster does not exist, and only if someone looks.

## Assessment

Triggered here by an old volume meeting a new binary — a state file written by
one version and read by another. That is not a supported upgrade path and may
never occur in production.

What is worth keeping is the shape: **an unreadable vote/membership file
degrades to "silently not a cluster member" rather than to a loud failure.**
For a consensus layer, refusing to start is almost certainly the better
behaviour — a node that cannot establish who it is should not pretend to be
healthy. An operator watching health checks sees three green nodes and no
quorum.

Fixing it properly means deciding between:

- **Fail fast**: treat an unreadable `raft_state.bin` alongside a non-empty log
  as fatal, and say what to do (wipe the raft dir to re-bootstrap, or restore
  the file).
- **Report degraded**: keep starting, but have `/admin/health` report
  not-ready while the node is a voter-less Learner with a non-empty log, so
  orchestration notices.

The second is probably right — a data-plane node that can still serve reads
is not useless — but either beats a green health check on a node that is not
in the cluster.

## Workaround

Remove the per-node raft volumes and let the cluster bootstrap clean:

```
docker compose --profile raft-cluster stop  veda-1 veda-2 veda-3
docker compose --profile raft-cluster rm -f veda-1 veda-2 veda-3
docker volume rm project-mdb_veda-{1,2,3}-data     # NOT minio/nats
docker compose --profile raft-cluster up -d veda-1 veda-2 veda-3
```

After that the cluster formed immediately (node-1 leader at term 3), and both
smokes passed against current `main`:

- `raft-partition-smoke.sh` — leader killed, node-3 elected, restarted node
  rejoined via WAL replay, all three converged at `last_applied_index=38`.
- `raft-snapshot-soak.sh` — follower's raft dir wiped entirely, rebuilt from
  the leader over Tonic, caught up (282 ≥ 266 target), all three converged.

Both had existed since 2026-05-19 with no record of a run against anything
current; the image on the host predated them by three months.
