"""
Test 5: Multi-Target Tracker Only
"""

import numpy as np

from sentinel.tracking.kalman_tracker import MultiTargetTracker


def test_tracker():
    print("=" * 60)
    print("TEST: Multi-Target Tracker")
    print("=" * 60)

    # Initialize tracker
    print("\n1. Initializing tracker...")
    tracker = MultiTargetTracker()
    print("   Tracker initialized")

    # Add first track
    print("\n2. Adding first target...")
    measurements = [np.array([100.0, 200.0, 50.0])]
    tracker.update_tracks(measurements, event_types=["launch"])
    tracks = tracker.get_active_tracks()
    print(f"   Active tracks: {len(tracks)}")
    print(f"   Track 0 position: {tracks[0].position}")

    # Update first track
    print("\n3. Updating first target...")
    measurements = [np.array([110.0, 205.0, 52.0])]
    tracker.update_tracks(measurements, event_types=["launch"])
    tracks = tracker.get_active_tracks()
    print(f"   Active tracks: {len(tracks)}")
    print(f"   Track 0 position: {tracks[0].position}")

    # Add second track
    print("\n4. Adding second target...")
    measurements = [np.array([120.0, 210.0, 54.0]), np.array([500.0, 600.0, 100.0])]
    tracker.update_tracks(measurements, event_types=["launch", "explosion"])
    tracks = tracker.get_active_tracks()
    print(f"   Active tracks: {len(tracks)}")
    for track in tracks:
        print(f"   Track {track.track_id}: {track.event_type} at {track.position}")

    print("\n" + "=" * 60)
    print("Multi-target tracker test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_tracker()
