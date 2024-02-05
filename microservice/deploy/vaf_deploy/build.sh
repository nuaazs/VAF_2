#!/bin/bash
# Change to zsh (already installed by apt)
echo "******************************************Change to zsh ... ******************************************"
chsh -s $(which zsh) <<< "Y"
# /usr/bin/zsh
sh -c "$(wget -O- https://gitee.com/pocmon/ohmyzsh/raw/master/tools/install.sh)" --unattended  <<< "Y"

# Download zsh plugins
ZSH_CUSTOM="/root/.oh-my-zsh/custom"
mkdir -p $ZSH_CUSTOM
git clone https://gitee.com/asddfdf/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
git clone https://gitee.com/chenweizhen/zsh-autosuggestions.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

# Change zsh config
cat <<EOF >> tmp.txt
ZSH_CUSTOM=/root/.oh-my-zsh/custom
source \$ZSH_CUSTOM/plugins/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh
ZSH_THEME="avit"
plugins=(
    git
    docker
    zsh-autosuggestions  # autosuggestions
    zsh-syntax-highlightling
)
export HOMEBREW_NO_AUTO_UPDATE=true                     # no update when use brew
export DISABLE_AUTO_UPDATE="true"
source \$ZSH_CUSTOM/plugins/zsh-autosuggestions/zsh-autosuggestions.zsh
source \$ZSH_CUSTOM/plugins/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh
alias pp='fzf --preview "[[ \$(file --mime {}) =~ binary ]] && echo {} is a binary file || (highlight -O ansi -l {} || coderay {} || rougify {} || cat {}) 2> /dev/null | head -500"'
alias oo='fzf --preview "[[ \$(file --mime {}) =~ binary ]] && echo {} is a binary file || (highlight -O ansi -l {} || coderay {} || rougify {} || tac {}) 2> /dev/null | head -500"'  # flashback
export FZF_DEFAULT_COMMAND='fdfind --type file'
export FZF_CTRL_T_COMMAND=\$FZF_DEFAULT_COMMAND
export FZF_ALT_C_COMMAND="fdfind -t d . "
export HISTSIZE=1000000000
export SAVEHIST=\$HISTSIZE
setopt EXTENDED_HISTORY
EOF

cat tmp.txt >> /root/.zshrc
rm tmp.txt

# resource zshrc
source /root/.zshrc




# Generate requirements.txt
echo "**************************************Generate requirements.txt**************************************"
cat <<EOF >> requirements.txt
Flask==2.2.5
funasr==0.6.5
gevent==23.9.1
gradio==3.44.4
gunicorn==21.2.0
hdbscan==0.8.33
huggingface-hub==0.16.4
jmpy3==1.0.6
loguru==0.7.1
matplotlib==3.7.2
minio==7.1.16
modelscope==1.6.1
nltk==3.8.1
opencv-python==4.8.0.76
oss2==2.18.2
paddlenlp==2.5.2
paddlepaddle==2.4.2
phone==0.4.3
Pillow==10.0.0
pyinstaller==5.13.2
PyMySQL==1.1.0
redis==5.0.1
soundfile==0.12.1
speechbrain==0.5.15
TextGrid==1.5
tqdm==4.66.1
uvicorn==0.23.2
wget==3.2
EOF

# Install packages
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple