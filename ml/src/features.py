import numpy as np

def _pairwise_deltas(times):
    if len(times) < 2:
        return np.array([])
    return np.diff(np.array(times))

def keystroke_features(keystrokes):
    downs, ups = [], []
    backspaces = 0
    for ev in keystrokes:
        if ev.get('type') == 'down':
            downs.append(ev['t'])
            if ev.get('key') == 'Backspace':
                backspaces += 1
        elif ev.get('type') == 'up':
            ups.append(ev['t'])

    ks_count = len(keystrokes)
    if len(downs) == 0:
        return {
            'ks_count': 0,
            'ks_rate': 0.0,
            'dwell_mean': 0.0, 'dwell_std': 0.0,
            'flight_mean': 0.0, 'flight_std': 0.0,
            'backspace_rate': 0.0,
            'digraph_mean': 0.0, 'digraph_std': 0.0
        }

    # typing speed
    span = (max(keystrokes, key=lambda x: x['t'])['t'] - min(keystrokes, key=lambda x: x['t'])['t']) / 1000.0
    span = span if span > 0 else 1.0
    ks_rate = ks_count / span

    # dwell time (key hold time)
    n = min(len(downs), len(ups))
    dwell = (np.array(ups[:n]) - np.array(downs[:n]))
    # flight time (time between key presses)
    flight = _pairwise_deltas(downs)

    backspace_rate = backspaces / max(1, len(downs))

    return {
        'ks_count': float(ks_count),
        'ks_rate': float(ks_rate),
        'dwell_mean': float(np.mean(dwell)) if dwell.size else 0.0,
        'dwell_std': float(np.std(dwell)) if dwell.size else 0.0,
        'flight_mean': float(np.mean(flight)) if flight.size else 0.0,
        'flight_std': float(np.std(flight)) if flight.size else 0.0,
        'backspace_rate': float(backspace_rate),
        'digraph_mean': float(np.mean(flight)) if flight.size else 0.0,
        'digraph_std': float(np.std(flight)) if flight.size else 0.0
    }

def extract_features(events):
    return keystroke_features(events.get('keystrokes', []))
