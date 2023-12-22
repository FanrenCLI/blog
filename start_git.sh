#！bin/bash
source /etc/profile
source ~/.bash_profile
log_path=git_log.txt
echo "begin at `date`" >$log_path
echo "---- remote status ---------" >> $log_path
git remote show origin >> $log_path
flag=1
for line in `sed -n "/local out of date/p" $log_path`
do  
flag=0                             
done
if (($flag == 0))
then
echo "有文件更新~~~"
git pull
hexo g
pm2 restart all
else
echo "github很干净哟"
fi