#!/usr/bin/env python3
"""Root-only CPU preparation from a freshly qualified independent producer."""
import argparse
from owned_layer3_qsa_control_v4 import prepare


def main():
    from qualify_owned_layer3_qsa_control_v4 import source_binding
    args = argparse.ArgumentParser()
    args.add_argument('--own-producer-root', required=True)
    args.add_argument('--output', required=True)
    options = args.parse_args()
    source_binding()
    prepare(options.own_producer_root, options.output)


if __name__ == '__main__':
    main()
