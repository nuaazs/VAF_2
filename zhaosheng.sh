# pip conda apt
alias pi='pip install $1'
alias doubanpi='pip install $1 -i https://pypi.doubanio.com/simple/  --trusted-host pypi.doubanio.com'
alias pu='pip uninstall $1'
alias install='sudo apt-get install -y $1'
alias upgrade='sudo apt-get upgrade'
alias update='sudo apt-get update'

# ubuntu
alias c='clear'
alias gpu='nvidia-smi'
alias watchgpu="watch -n 1 nvidia-smi"
alias hg='history | grep $1'
alias ip='ifconfig'
alias howmany='ls -l | grep -v ^l | wc -l'
alias diff="icdiff"
alias 777='sudo chmod 777 $1'
alias network='nm-connection-editor'

# set history format
export HISTTIMEFORMAT="%F %T `whoami` "
alias ls='ls -hl --color=auto'
export HISTIGNORE="pwd:ls:ls -ltr:ls -h"
export HISTSIZE=2000

# bashrc
alias bashrc='vim ~/.bashrc'
alias sourcerc='source ~/.bashrc'

# proxy
alias proxy='export http_proxy=http://192.168.3.28:7890; export https_proxy=http://192.168.3.28:7890;'
alias socks5='export all_proxy="socks5://127.0.0.1:10810" && export ALL_PROXY="socks5://127.0.0.1:10810" && git config --global https.proxy "socks5://127.0.0.1:10810"'
#alias unset_proxy='unset all_proxy; unset ALL_PROXY; git config --global --unset https.proxy; git config --global --unset http.proxy && git config --unset http.proxy && git config --unset https.proxy'
alias unset_proxy='export http_proxy=""; export https_proxy="";'
# open
alias open_root='sudo nautilus'
alias open='xdg-open'

# Python & Conda
alias nb='ipython notebook --no-browser --port=8899 --allow-root'
alias jb='ipython notebook --no-browser --allow-root'
alias jl='jupyter lab --no-browser --allow-root'

# Tmux
alias tl='tmux ls'
alias ta='tmux attach-session -d -t'
alias tn='tmux new-session -s'
alias tk='tmux kill-session -t'
alias sv='tmux split-window'
alias sh='tmux split-window -h'

# x11
alias xau='cp /home/iint/.Xauthority /root/'

# SSH
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=80
export ANTS_RANDOM_SEED=3

mkcd ()
{
	mkdir -p -- "$1" &&
	  cd -P -- "$1"
}

# VScode
alias code='code . --user-data-dir'
alias docker='sudo docker'

# shorthand for vim
alias v="vim"

# setting preferred options on ls
alias ls='ls -lhF'

# prompt user if overwriting during copy
alias cp='cp -i'

# prompt user when deleting a file
alias rm='rm -i'

# always print in human readable form
alias df="df -h"

alias lsi="ls ./ --ignore"
alias ff="find ./ -name"
alias disk="sudo fdisk -l;df -hl;"
alias sb="cd /home/zhaosheng/speechbrain/recipes/VoxCeleb/SpeakerRec"
alias howbig="du -sh * | sort -n"
alias dev="conda activate server_dev"


export PATH=$PATH:/home/zhaosheng/bin:/home/zhaosheng/VAF_UTILS/utils/examples/ex16_vad_cpp
# redis mysql
alias start_redis='sudo systemctl start redis-server'
alias start_mysql='sudo systemctl start mysql.service'
# alias docker_rmi="sudo docker rmi $(docker images | grep '^<none>' | awk '{print $3}')"
alias dp='sudo docker ps -a'
alias di='sudo docker images'
NCCL_P2P_DISABLE=1
alias ca='conda activate $1'
alias io='iostat -x 1 10'


# GUM
# alias gum_choose="cat $1 | gum choose --limit 999 --height 15 "


# temp path
alias server="/home/zhaosheng/asr_damo_websocket/online/microservice"
alias utils='/home/zhaosheng/VAF_UTILS/utils/examples'
bot() {
    message="$1"    # 获取待发送的消息
    shift           # 将命令行参数左移，去掉第一个参数
    cmd="$@"        # 获取剩余的参数作为命令    
    echo "$cmd"
    wechat $(eval "$cmd")
}
export DGUARD_DEPLOY_ROOT=/VAF/model_deploy
export DGUARD_ROOT=/VAF/train/dguard
export DGUARD_TEST_ROOT=/VAF/model_test
export DGUARD_EMBEDDING_TEST_ROOT=/VAF/model_embedding_test

