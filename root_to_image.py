import os
import csv
import argparse

import uproot
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Image parameters : like oc
# ============================================================

X_MIN = -2000
X_MAX =  2000

Y_MIN = -2000
Y_MAX =  2000

FIGSIZE = (2, 2)
DPI = 50

POINT_SIZE = 1

IMG_WIDTH = 100
IMG_HEIGHT = 100


# ============================================================
# ROOT - PNG
# ============================================================

def convert_root_to_images(root_file, output_dir, particle_id,
                           momentum=30.0, eta=2.5, phi=np.pi/2):

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nOpening: {root_file}")

    with uproot.open(root_file) as f:

        tree = f["events"]

        n_events = tree.num_entries

        print(f"Number of events: {n_events}")

        # Read the dRICH hit positions
        arrays = tree.arrays(
            [
                "DRICHHits.position.x",
                "DRICHHits.position.y"
            ],
            library="ak"
        )

        x_hits = arrays["DRICHHits.position.x"]
        y_hits = arrays["DRICHHits.position.y"]

        print("Successfully read dRICH hit positions.")

        
        # data file
        # ----------------------------------------------------

        metadata_file = os.path.join(output_dir, "metadata.csv")

        with open(metadata_file, "w", newline="") as csvfile:

            writer = csv.writer(csvfile)

            writer.writerow([
                "image",
                "particle_id",
                "momentum",
                "eta",
                "phi",
                "event"
            ])

            # ------------------------------------------------
            # Process every event
            # ------------------------------------------------

            for event in range(n_events):

                x = np.asarray(x_hits[event])
                y = np.asarray(y_hits[event])

                # --------------------------------------------
                # Same plotting procedure as your oc
                # --------------------------------------------

                plt.figure(figsize=(2, 2))

                plt.scatter(
                    x,
                    y,
                    s=1,
                    color="blue"
                )

                plt.axis("off")

                plt.xlim(-2000, 2000)
                plt.ylim(-2000, 2000)

                filename = f"event_{event:04d}.png"

                filepath = os.path.join(
                    output_dir,
                    filename
                )

                plt.savefig(
                    filepath,
                    dpi=50,
                )

                plt.close()
                # --------------------------------------------
                # Save real data separately
                # --------------------------------------------

                writer.writerow([
                    filename,
                    particle_id,
                    momentum,
                    eta,
                    phi,
                    event
                ])

                if (event + 1) % 10 == 0 or event == n_events - 1:
                    print(
                        f"Processed {event + 1}/{n_events} events"
                    )

    print(f"\nImages saved in:")
    print(output_dir)

    print(f"Metadata saved in:")
    print(metadata_file)


# ============================================================
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert dRICH hits from EDM4hep ROOT file to PNG images."
    )

    parser.add_argument(
        "root_file",
        help="Input EDM4hep ROOT file"
    )

    parser.add_argument(
        "output_dir",
        help="Directory where images will be saved"
    )

    parser.add_argument(
        "--pid",
        type=int,
        required=True,
        help="Particle PDG ID"
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=30.0,
        help="Particle momentum in GeV/c"
    )

    parser.add_argument(
        "--eta",
        type=float,
        default=2.5,
        help="Particle pseudorapidity"
    )

    parser.add_argument(
        "--phi",
        type=float,
        default=np.pi/2,
        help="Particle azimuthal angle"
    )

    args = parser.parse_args()

    convert_root_to_images(
        root_file=args.root_file,
        output_dir=args.output_dir,
        particle_id=args.pid,
        momentum=args.momentum,
        eta=args.eta,
        phi=args.phi
    )
