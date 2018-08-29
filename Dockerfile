FROM registry.cn-hangzhou.aliyuncs.com/denverdino/tensorflow:1.5.0-gpu-py3

LABEL maintainer "raven4752 raven4752@foxmail.com>"


RUN apt-get update &&\
    apt-get install -y openssh-server
COPY requirements.txt .
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

RUN mkdir /var/run/sshd
RUN echo 'root:root'|chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
CMD ["/usr/sbin/sshd", "-D"]
EXPOSE 22
EXPOSE 8888
