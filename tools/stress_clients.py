#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field


def normalize_base_url(raw: str) -> str:
    value = raw.strip().rstrip("/")
    if not value:
        raise ValueError("base-url cannot be empty")
    if not value.endswith("/v1"):
        value = f"{value}/v1"
    return value


def parse_sweep(raw: str | None, default_clients: int) -> list[int]:
    if not raw:
        return [default_clients]
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 1:
            raise ValueError("sweep values must be >= 1")
        out.append(value)
    if not out:
        raise ValueError("sweep cannot be empty")
    return out


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    items = sorted(values)
    pos = (len(items) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(items) - 1)
    if lo == hi:
        return items[lo]
    ratio = pos - lo
    return items[lo] * (1.0 - ratio) + items[hi] * ratio


@dataclass
class RunMetrics:
    started_at: float
    finished_at: float = 0.0
    requests_total: int = 0
    requests_ok: int = 0
    requests_failed: int = 0
    bytes_received: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    ttft_ms: list[float] = field(default_factory=list)
    errors: dict[str, int] = field(default_factory=dict)
    per_client_requests: dict[int, int] = field(default_factory=dict)
    per_client_latency_ms_total: dict[int, float] = field(default_factory=dict)
    per_client_ttft_ms_total: dict[int, float] = field(default_factory=dict)

    def add_error(self, label: str) -> None:
        self.errors[label] = self.errors.get(label, 0) + 1


def http_get_models(base_url: str, timeout_seconds: float) -> list[str]:
    url = f"{base_url}/models"
    req = urllib.request.Request(url=url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        payload = resp.read()
    data = json.loads(payload.decode("utf-8"))
    cards = data.get("data", [])
    return [item.get("id", "") for item in cards if isinstance(item, dict)]


def request_once(
    *,
    endpoint: str,
    payload_bytes: bytes,
    timeout_seconds: float,
    stream: bool,
) -> tuple[str, int, float, float | None]:
    started = time.perf_counter()
    status = "ok"
    received_bytes = 0
    ttft_ms: float | None = None
    try:
        req = urllib.request.Request(
            url=endpoint,
            method="POST",
            data=payload_bytes,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            if resp.status != 200:
                status = f"http_{resp.status}"
            if stream:
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    received_bytes += len(line)
                    if not line.startswith(b"data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw:
                        continue
                    if raw == b"[DONE]":
                        break
                    try:
                        payload = json.loads(raw.decode("utf-8"))
                    except Exception:
                        continue
                    if isinstance(payload, dict):
                        err = payload.get("error")
                        if isinstance(err, dict) and err.get("message"):
                            status = f"stream_error:{str(err.get('message'))[:120]}"
                            break
                        choice = (payload.get("choices") or [{}])[0]
                        token = choice.get("text")
                        if token is None:
                            token = (choice.get("delta") or {}).get("content")
                        if token and ttft_ms is None:
                            ttft_ms = (time.perf_counter() - started) * 1000.0
            else:
                raw = resp.read()
                received_bytes = len(raw)
    except urllib.error.HTTPError as exc:
        status = f"http_{exc.code}"
        try:
            body = exc.read().decode("utf-8", errors="ignore")
            if body:
                status = f"{status}:{body[:140]}"
        except Exception:
            pass
    except urllib.error.URLError as exc:
        status = f"url_error:{exc.reason}"
    except TimeoutError:
        status = "timeout"
    except Exception as exc:  # noqa: BLE001
        status = f"error:{type(exc).__name__}"

    total_ms = (time.perf_counter() - started) * 1000.0
    return status, received_bytes, total_ms, ttft_ms


def run_once(
    *,
    base_url: str,
    model: str,
    clients: int,
    duration_seconds: float,
    timeout_seconds: float,
    max_tokens: int,
    temperature: float,
    prompt: str,
    think_time_ms: int,
    stream: bool,
    report_interval_seconds: float,
) -> RunMetrics:
    endpoint = f"{base_url}/completions"
    stop_at = time.monotonic() + duration_seconds
    done = threading.Event()
    lock = threading.Lock()
    metrics = RunMetrics(started_at=time.monotonic())

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    payload_bytes = json.dumps(payload).encode("utf-8")

    def worker(worker_id: int) -> None:
        while not done.is_set():
            now = time.monotonic()
            if now >= stop_at:
                done.set()
                return
            status, received_bytes, latency_ms, ttft_ms = request_once(
                endpoint=endpoint,
                payload_bytes=payload_bytes,
                timeout_seconds=timeout_seconds,
                stream=stream,
            )
            with lock:
                metrics.requests_total += 1
                metrics.latencies_ms.append(latency_ms)
                metrics.bytes_received += received_bytes
                metrics.per_client_requests[worker_id] = metrics.per_client_requests.get(worker_id, 0) + 1
                metrics.per_client_latency_ms_total[worker_id] = (
                    metrics.per_client_latency_ms_total.get(worker_id, 0.0) + latency_ms
                )
                if ttft_ms is not None:
                    metrics.ttft_ms.append(ttft_ms)
                    metrics.per_client_ttft_ms_total[worker_id] = (
                        metrics.per_client_ttft_ms_total.get(worker_id, 0.0) + ttft_ms
                    )
                if status == "ok":
                    metrics.requests_ok += 1
                else:
                    metrics.requests_failed += 1
                    metrics.add_error(status)
            if think_time_ms > 0:
                time.sleep(think_time_ms / 1000.0)

    last_report = time.monotonic()
    with ThreadPoolExecutor(max_workers=clients) as pool:
        for idx in range(clients):
            pool.submit(worker, idx)
        while not done.wait(timeout=0.2):
            if time.monotonic() >= stop_at:
                done.set()
                continue
            if report_interval_seconds > 0 and time.monotonic() - last_report >= report_interval_seconds:
                with lock:
                    total = metrics.requests_total
                    ok = metrics.requests_ok
                    fail = metrics.requests_failed
                    avg_lat = statistics.mean(metrics.latencies_ms) if metrics.latencies_ms else 0.0
                    p95_lat = percentile(metrics.latencies_ms, 0.95) if metrics.latencies_ms else 0.0
                    avg_ttft = statistics.mean(metrics.ttft_ms) if metrics.ttft_ms else 0.0
                elapsed = max(0.001, time.monotonic() - metrics.started_at)
                rps = total / elapsed
                print(
                    f"[live clients={clients}] req={total} ok={ok} fail={fail} "
                    f"rps={rps:.2f} avg={avg_lat:.1f}ms p95={p95_lat:.1f}ms "
                    f"ttft_avg={avg_ttft:.1f}ms"
                )
                last_report = time.monotonic()
    metrics.finished_at = time.monotonic()
    return metrics


def print_summary(clients: int, metrics: RunMetrics) -> None:
    duration = max(0.001, metrics.finished_at - metrics.started_at)
    rps = metrics.requests_total / duration
    ok_pct = (metrics.requests_ok / metrics.requests_total * 100.0) if metrics.requests_total else 0.0
    fail_pct = (metrics.requests_failed / metrics.requests_total * 100.0) if metrics.requests_total else 0.0
    avg = statistics.mean(metrics.latencies_ms) if metrics.latencies_ms else 0.0
    med = statistics.median(metrics.latencies_ms) if metrics.latencies_ms else 0.0
    p95 = percentile(metrics.latencies_ms, 0.95)
    p99 = percentile(metrics.latencies_ms, 0.99)
    avg_ttft = statistics.mean(metrics.ttft_ms) if metrics.ttft_ms else 0.0
    p95_ttft = percentile(metrics.ttft_ms, 0.95) if metrics.ttft_ms else 0.0
    mib = metrics.bytes_received / (1024 * 1024)

    print(f"\n=== Clients: {clients} ===")
    print(f"Duration: {duration:.1f}s")
    print(f"Total requests: {metrics.requests_total}")
    print(f"Success: {metrics.requests_ok} ({ok_pct:.1f}%)")
    print(f"Failed: {metrics.requests_failed} ({fail_pct:.1f}%)")
    print(f"Throughput: {rps:.2f} req/s")
    print(f"Latency ms: avg={avg:.1f} p50={med:.1f} p95={p95:.1f} p99={p99:.1f}")
    if metrics.ttft_ms:
        print(f"TTFT ms: avg={avg_ttft:.1f} p95={p95_ttft:.1f}")
    print(f"Payload received: {mib:.2f} MiB")
    if metrics.per_client_requests:
        print("Per-client avg latency (ms):")
        for client_id in sorted(metrics.per_client_requests):
            count = metrics.per_client_requests[client_id]
            if count < 1:
                continue
            avg_client = metrics.per_client_latency_ms_total.get(client_id, 0.0) / count
            ttft_total = metrics.per_client_ttft_ms_total.get(client_id, 0.0)
            avg_client_ttft = ttft_total / count if ttft_total > 0 else 0.0
            print(f"  - client {client_id}: req={count} avg={avg_client:.1f} ttft={avg_client_ttft:.1f}")
    if metrics.errors:
        print("Top errors:")
        for key, count in sorted(metrics.errors.items(), key=lambda item: item[1], reverse=True)[:5]:
            print(f"  - {count}x {key}")


def recommend_capacity(results: list[tuple[int, RunMetrics]]) -> None:
    stable = []
    for clients, metrics in results:
        duration = max(0.001, metrics.finished_at - metrics.started_at)
        rps = metrics.requests_total / duration if duration else 0.0
        error_rate = (metrics.requests_failed / metrics.requests_total) if metrics.requests_total else 1.0
        p95 = percentile(metrics.latencies_ms, 0.95)
        if error_rate <= 0.05:
            stable.append((clients, rps, p95, error_rate))
    if not stable:
        print("\nNo stable concurrency level found (<=5% errors).")
        return
    best = max(stable, key=lambda item: (item[0], item[1]))
    clients, rps, p95, error_rate = best
    print(
        f"\nRecommended concurrent clients on this GPU/model: {clients} "
        f"(error={error_rate * 100:.1f}%, throughput={rps:.2f} req/s, p95={p95:.1f}ms)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stress test concurrent clients against a single OpenAI-compatible /v1 model endpoint."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="Controller base URL (with or without /v1)")
    parser.add_argument("--model", required=True, help="Model name exactly as shown by GET /v1/models")
    parser.add_argument("--clients", type=int, default=1, help="Concurrent virtual clients")
    parser.add_argument("--sweep", default=None, help="Comma list of client counts, e.g. 1,2,4,8,12")
    parser.add_argument("--duration", type=float, default=60.0, help="Seconds per run")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout seconds")
    parser.add_argument("--max-tokens", type=int, default=128, help="max_tokens per request")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--prompt", default="Write a short paragraph about distributed systems.", help="Prompt text")
    parser.add_argument("--think-time-ms", type=int, default=0, help="Pause between requests per client")
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use streaming responses (default: true)",
    )
    parser.add_argument(
        "--report-interval",
        type=float,
        default=5.0,
        help="Print live aggregate stats every N seconds (0 disables live logs)",
    )
    parser.add_argument("--skip-model-check", action="store_true", help="Skip /v1/models availability check")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.clients < 1:
        raise SystemExit("--clients must be >= 1")
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens must be >= 1")
    if args.duration <= 0:
        raise SystemExit("--duration must be > 0")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be > 0")

    base_url = normalize_base_url(args.base_url)
    sweep = parse_sweep(args.sweep, args.clients)

    print(f"Target API: {base_url}")
    print(f"Model: {args.model}")
    print(f"Sweep: {sweep}")
    print(f"Duration per run: {args.duration:.1f}s")
    print(
        f"Request shape: max_tokens={args.max_tokens}, temperature={args.temperature}, "
        f"stream={args.stream}"
    )

    if not args.skip_model_check:
        try:
            models = http_get_models(base_url, timeout_seconds=args.timeout)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"Failed to query {base_url}/models: {exc}") from exc
        if args.model not in models:
            pretty = ", ".join(models) if models else "(none)"
            raise SystemExit(f"Model '{args.model}' not available in /v1/models. Available: {pretty}")

    results: list[tuple[int, RunMetrics]] = []
    for clients in sweep:
        metrics = run_once(
            base_url=base_url,
            model=args.model,
            clients=clients,
            duration_seconds=args.duration,
            timeout_seconds=args.timeout,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            prompt=args.prompt,
            think_time_ms=args.think_time_ms,
            stream=args.stream,
            report_interval_seconds=args.report_interval,
        )
        results.append((clients, metrics))
        print_summary(clients, metrics)

    recommend_capacity(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
