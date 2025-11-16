def main():
    from scripts import generate_reverb, run_baselines, evaluate_methods, export_demos

    print("[run_all] Step 1: generate_reverb ...")
    generate_reverb.main()

    print("[run_all] Step 2: run_baselines ...")
    run_baselines.main()

    try:
        from scripts import train_tiny_unet, run_tiny_unet

        print("[run_all] Step 3: train_tiny_unet ...")
        train_tiny_unet.main()

        print("[run_all] Step 4: run_tiny_unet ...")
        run_tiny_unet.main()
    except ImportError as e:
        print("[run_all] PyTorch not available, skipping tiny UNet: ", e)

    print("[run_all] Step 5: evaluate_methods ...")
    evaluate_methods.main()

    print("[run_all] Step 6: export_demos ...")
    export_demos.main()

    print("[run_all] All steps finished.")


if __name__ == "__main__":
    main()
