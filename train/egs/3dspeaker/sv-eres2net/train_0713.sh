#!/bin/bash
. ./run.sh || wechat echo "eres2net train error (phone)"
wechat echo "eres2net train done (phone)"
. ./run_3dspeaker_raw.sh || wechat echo "eres2net train error (phone)"
wechat echo "eres2net train done (raw)"

