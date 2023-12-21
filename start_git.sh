#！bin/bash
source /etc/profile
source ~/.bash_profile
log_path=git_log.txt
echo "begin at `date`" >$log_path
echo "---- remote status ---------" >> $log_path
git remote show origin >> $log_path
flag=1
if sed -n "/local out of date/p" $log_path
then
flag=0
fi
if (($flag == 0))
then
git pull
hexo g
pm2 restart all
else
echo "github很干净哟"
fi