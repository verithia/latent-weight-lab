#!/usr/bin/env bash
# Detached-run watchdog.  It observes only this prep process group, its status,
# and output bin sizes; it does not inspect GPUs or unrelated processes.
set -euo pipefail
PGID="${1:?usage: $0 PGID STAGING_DIR WORKSPACE [log]}"; STAGE="${2:?}"; WORKSPACE="${3:?}"; LOG="${4:-$STAGE/watchdog.log}"
CALLBACK_URL="${PREP_CALLBACK_URL:-http://127.0.0.1:8766/send-opencode-test}"
last_bytes=-1; stagnant=0; sent20=0; sent50=0; sent80=0
notify() { local msg="$1" payload; printf '%s %s\n' "$(date -Is)" "$msg" >>"$LOG"; payload=$(python3 -c 'import json,sys; print(json.dumps({"chat_id":"oc_fa5c2ec0190c9444cce960125eafff50","text":sys.argv[1]}))' "$msg"); [[ -z "$CALLBACK_URL" ]] || env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY curl --noproxy '*' --connect-timeout 10 --max-time 20 -fsS -X POST -H 'Content-Type: application/json' --data-binary "$payload" "$CALLBACK_URL" >/dev/null || true; }
while kill -0 -- "-$PGID" 2>/dev/null; do
  status="$STAGE/prep-status.json"; phase=writing; done_tokens=0; target=1
  if [[ -f "$status" ]]; then read -r phase done_tokens target < <(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("phase","writing"),d.get("committed_tokens",0),d.get("target_tokens",1))' "$status"); fi
  # Growth deliberately counts token bins only, not logs, caches, or state files.
  bytes=$(python3 -c 'import pathlib,sys; print(sum(p.stat().st_size for p in pathlib.Path(sys.argv[1]).rglob("*.bin*") if p.is_file()))' "$STAGE")
  pct=$(( done_tokens * 100 / target ));
  if (( pct >= 20 && !sent20 )); then notify "FineWeb-Edu prep ${PGID}: 20%"; sent20=1; fi
  if (( pct >= 50 && !sent50 )); then notify "FineWeb-Edu prep ${PGID}: 50%"; sent50=1; fi
  if (( pct >= 80 && !sent80 )); then notify "FineWeb-Edu prep ${PGID}: 80%"; sent80=1; fi
  if (( bytes > last_bytes )); then stagnant=0; last_bytes=$bytes; else stagnant=$((stagnant + 60)); fi
  ws=$(du -sk "$WORKSPACE" 2>/dev/null | cut -f1 || printf 0); ws=$((ws * 1024))
  avail=$(df -Pk "$WORKSPACE" | python3 -c 'import sys; print(int(sys.stdin.readlines()[-1].split()[3])*1024)')
  if (( ws >= 192*1024*1024*1024 || avail < 64*1024*1024*1024 )); then notify "FineWeb-Edu prep ${PGID}: SPACE DANGER; stopping"; kill -- "-$PGID" || true; break; fi
  if (( stagnant >= 900 )); then notify "FineWeb-Edu prep ${PGID}: no bin growth for 15 minutes"; fi
  if (( stagnant >= 2700 )) && [[ "$phase" != verification ]]; then notify "FineWeb-Edu prep ${PGID}: no growth for 45 minutes; stopping"; kill -- "-$PGID" || true; break; fi
  sleep 60
done
notify "FineWeb-Edu prep ${PGID}: process group exited"
