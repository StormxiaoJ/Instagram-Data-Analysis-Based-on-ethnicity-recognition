FROM centos:6.9


USER root
RUN echo root:test | chpasswd

RUN yum -y update 
RUN yum -y install -y which wget net-tools rsync && \
    yum clean all && \
    cp /usr/share/zoneinfo/Europe/Paris /etc/localtime 

# jdk
ENV JAVA_HOME=/usr/lib/jvm/java-openjdk
ENV PATH=$PATH:$JAVA_HOME/bin:.
RUN yum install -y java-1.8.0-openjdk-devel && yum clean all

# SSH
RUN yum install -y openssl openssh-server openssh-clients && \
    yum clean all && \
    ssh-keygen -t rsa -f ~/.ssh/id_rsa -P '' && \
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
   
ADD config/ssh_config /root/.ssh/config
RUN chmod 600 /root/.ssh/config && \
    chown root:root /root/.ssh/config

CMD [ "sh", "-c", "service sshd start; bash"]