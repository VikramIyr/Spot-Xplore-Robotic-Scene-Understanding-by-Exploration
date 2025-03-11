#!/bin/bash
# sudo route add -net 192.168.50.0/24 netmask 192.168.1.243
sudo ip route add 192.168.50.0/24 via 192.168.1.243
