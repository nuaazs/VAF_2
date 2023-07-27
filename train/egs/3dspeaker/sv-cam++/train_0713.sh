#!/bin/bash
. ./run.sh || wechat echo "cam++ train error (phone)"
wechat echo "cam++ train done (phone)"
. ./run_3dspeaker_raw.sh || wechat echo "cam++ train error (phone)"
wechat echo "cam++ train done (raw)"

