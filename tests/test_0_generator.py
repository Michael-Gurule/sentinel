"""
Test 0: Verify Signal Generator Works
"""

from sentinel.models.signal_generator import OPIRSignalGenerator


def test_generator():
    print("=" * 60)
    print("TEST: Signal Generator")
    print("=" * 60)

    # Initialize generator
    print("\n1. Initializing generator...")
    generator = OPIRSignalGenerator()
    print("   Generator initialized")

    # Check attributes
    print("\n2. Generator attributes:")
    attrs = [a for a in dir(generator) if not a.startswith("_")]
    for attr in attrs:
        try:
            value = getattr(generator, attr)
            if not callable(value):
                print(f"   {attr}: {value}")
        except:
            pass

    # Generate signal
    print("\n3. Generating launch signal...")
    try:
        signal = generator.generate_launch_signature(start_time=2.0)
        print(f"   Signal shape: {signal.shape}")
        print(f"   Signal min: {signal.min():.3f}")
        print(f"   Signal max: {signal.max():.3f}")
        print(f"   Signal mean: {signal.mean():.3f}")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Generator test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_generator()
