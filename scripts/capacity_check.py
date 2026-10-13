"""Header-only EXL3 capacity estimate using the installed pinned TensorFold.

No weights are allocated. --live initialises CUDA to read the selected card's
free memory, then checks the unmodified upstream admission plan. Passing this
check is NOT a cold-start, inference, peak-memory or quality qualification.
"""
import argparse
import json
from pathlib import Path

GIB = 1024 ** 3


def require_admission(result):
    if result['fitting'] < result['requested']:
        raise ValueError('Upstream memory admission refuses the requested context; nothing was started.')
    host = result['host_budget_bytes']
    if host is None or host < result['staging_bytes']:
        raise ValueError('Insufficient or unknown host staging memory; nothing was started.')


def estimate(model_dir, context, total, budget, host_budget=None):
    from tensorfold.cuda.capacity import estimate_weights, headers, make_plan
    from tensorfold.cuda.geometry import gdn_geometry, prompt_rows, prompt_row_bytes
    from tensorfold.families.qwen3_5.cuda.exl3_load import admission

    config = json.loads((model_dir / 'config.json').read_text())
    text = config.get('text_config', config)
    quant = config.get('quantization_config', text.get('quantization_config', {}))
    if str(quant.get('quant_method', '')).lower() != 'exl3':
        raise ValueError('This profile is for the pinned EXL3 checkpoint only.')
    chunk = prompt_rows(total, prompt_row_bytes(text))
    geometry, transform = admission(
        lambda t: gdn_geometry(t, 1, 12, rows=12, prompt=chunk, evicts=True),
        layer_major=True)
    weights = estimate_weights(model_dir, transform)
    geom = geometry(text)
    plan = make_plan(int(text['max_position_embeddings']), context, True, budget, weights, geom)
    return {
        'scope': 'Header/source estimate only; no model cold start or GPU inference',
        'tensor_headers': len(headers(model_dir)),
        'requested': context, 'total_bytes': total, 'budget_bytes': budget,
        'host_budget_bytes': host_budget, 'prompt_chunk_rows': chunk,
        'resident_bytes': weights.resident, 'staging_bytes': weights.staging,
        'startup_peak_bytes': weights.resident + weights.staging,
        'cache_workspace_bytes': geom.needed(context),
        'serving_peak_bytes': weights.resident + geom.needed(context),
        'estimated_peak_bytes': weights.resident + max(weights.staging, geom.needed(context)),
        'fitting': plan.fitting, 'largest': plan.largest, 'plan': plan.receipt(context),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--context', type=int, default=8192)
    parser.add_argument('--total-gib', type=float, default=24)
    parser.add_argument('--budget-gib', type=float, default=20)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.context <= 8192 or not 0 < args.budget_gib <= args.total_gib:
        parser.error('context must be 1..8192 and budget must be positive and no greater than total')
    if args.live:
        import torch
        from tensorfold.cuda.capacity import available_bytes, host_stream_bytes
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise SystemExit('Expose exactly one CUDA GPU to the capacity check.')
        if torch.cuda.get_device_capability(0) < (8, 0):
            raise SystemExit('EXL3 profile requires Ampere or newer; nothing was started.')
        props = torch.cuda.get_device_properties(0)
        if getattr(props, 'is_integrated', False):
            raise SystemExit('Use the Spark profile for unified-memory hardware.')
        free, total = map(int, torch.cuda.mem_get_info())
        result = estimate(args.model, args.context, total, available_bytes(torch), host_stream_bytes())
        result['gpu_name'] = torch.cuda.get_device_name(0)
        result['compute_capability'] = list(torch.cuda.get_device_capability(0))
        result['free_bytes_after_cuda_init'] = free
    else:
        result = estimate(args.model, args.context, int(args.total_gib * GIB), int(args.budget_gib * GIB))
    print(json.dumps(result, indent=2), flush=True)
    if args.live:
        try:
            require_admission(result)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        print('Header admission passed. Cold start, inference and peak-memory qualification remain required.')


if __name__ == '__main__':
    main()
