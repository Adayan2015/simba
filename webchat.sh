#!/bin/sh
param=$1
cd `dirname "$0"`

start()
{
    py=`ps aux | grep -i "webchat/index" | grep -v grep | awk '{print $2}'`
    if [ ! -n "$py" ]; then
        nohup python webchat/index.py &
        echo "Start: nohup python webchat/index.py &"
    else
        echo "webchat/index exists."
    fi
    ps aux|grep -i "webchat/index"

}

stop()
{
    py=`ps aux | grep -i "webchat/index" | grep -v grep | awk '{print $2}'`
    echo $py | xargs kill -9

    for pid in $py; do
        if echo $pid | egrep -q '^[0-9]+$'; then
            echo "webchat/index pid $pid kill"
        else
            echo "$pid is not a webchat/index pid"
        fi
    done
    ps aux|grep -i "webchat/index"
}

case $param in
    'start')
        start;;
    'stop')
        stop;;
    'restart')
        stop
        start;;
    *)
        echo "Usage: ./webchat.sh start|stop|restart";;
esac
