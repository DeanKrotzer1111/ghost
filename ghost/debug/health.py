"""Ghost Health Check — validates all services are reachable."""
import asyncio
import httpx
import sys


SERVICES = {
    "Ghost API": "http://localhost:8080/health",
    "Qwen MLX": "http://localhost:8081/v1/models",
    "Dashboard": "http://localhost:3000",
}


async def check_health():
    results = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in SERVICES.items():
            try:
                r = await client.get(url)
                results[name] = f"OK ({r.status_code})"
            except Exception as e:
                results[name] = f"FAIL ({type(e).__name__})"

    print("GHOST HEALTH CHECK")
    print("=" * 40)
    for name, status in results.items():
        icon = "+" if "OK" in status else "X"
        print(f"  [{icon}] {name}: {status}")

    all_ok = all("OK" in s for s in results.values())
    print(f"\n{'All systems operational.' if all_ok else 'Some services unavailable.'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(check_health()))
