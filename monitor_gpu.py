#!/usr/bin/env python3
import time
import torch
import psutil
import argparse
from datetime import datetime


def format_bytes(bytes_val):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def get_gpu_memory():
    if not torch.cuda.is_available():
        return None
    
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    total = torch.cuda.get_device_properties(0).total_memory
    
    return {
        'allocated': allocated,
        'reserved': reserved,
        'total': total,
        'free': total - allocated,
        'utilization': (allocated / total) * 100
    }


def get_cpu_memory():
    mem = psutil.virtual_memory()
    return {
        'total': mem.total,
        'available': mem.available,
        'used': mem.used,
        'utilization': mem.percent
    }


def print_stats(show_cpu=True):
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    print(f"\n[{timestamp}] Memory Stats")
    print("=" * 60)
    
    gpu_mem = get_gpu_memory()
    if gpu_mem:
        print(f"GPU Memory:")
        print(f"  Allocated:   {format_bytes(gpu_mem['allocated'])} ({gpu_mem['utilization']:.1f}%)")
        print(f"  Reserved:    {format_bytes(gpu_mem['reserved'])}")
        print(f"  Free:        {format_bytes(gpu_mem['free'])}")
        print(f"  Total:       {format_bytes(gpu_mem['total'])}")
        
        if gpu_mem['utilization'] > 90:
            print(f"  ⚠️  WARNING: GPU usage > 90%")
        elif gpu_mem['utilization'] > 80:
            print(f"  ⚠️  CAUTION: GPU usage > 80%")
    else:
        print("GPU: Not available")
    
    if show_cpu:
        cpu_mem = get_cpu_memory()
        print(f"\nCPU Memory:")
        print(f"  Used:        {format_bytes(cpu_mem['used'])} ({cpu_mem['utilization']:.1f}%)")
        print(f"  Available:   {format_bytes(cpu_mem['available'])}")
        print(f"  Total:       {format_bytes(cpu_mem['total'])}")
    
    print("=" * 60)


def monitor_loop(interval=2, show_cpu=True):
    print("GPU Memory Monitor")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        while True:
            print_stats(show_cpu=show_cpu)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


def main():
    parser = argparse.ArgumentParser(description="Monitor GPU and CPU memory usage")
    parser.add_argument(
        "--interval",
        type=int,
        default=2,
        help="Update interval in seconds (default: 2)"
    )
    parser.add_argument(
        "--no-cpu",
        action="store_true",
        help="Don't show CPU memory stats"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print stats once and exit"
    )
    
    args = parser.parse_args()
    
    if args.once:
        print_stats(show_cpu=not args.no_cpu)
    else:
        monitor_loop(interval=args.interval, show_cpu=not args.no_cpu)


if __name__ == "__main__":
    main()
