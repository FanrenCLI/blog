# blog

## 安装运行教程

1. 从gitee上拉取master分支
2. 在本地安装nodeJS环境
3. 通过npm命令安装package.json文件中提到的所有依赖
   - `npm install` 后面不跟参数表示读取当前目录下的package.json中的信息并将读取到的所需依赖安装到本目录下面
   - `npm install -g (global) [jquery]`表示将jquery依赖安装在全局目录下面，一般为C盘中的位置
   - `npm install -s (save) [jquery]`表示将jquery依赖安装在本地目录并且加入到package.json中保存，
4. 下载玩所有依赖之后再目录中会出现node_modules
5. 此时根据`package.json`中的依赖，我们就会安装好hexo，就可以使用hexo进行操作了
6. 此外，想要在服务器中运行托管hexo还需要额外安装PM2插件
7. 推荐选择全局安装`npm install -g pm2`
8. 然后通过`pm2 start hexo.js`启动项目

## Hexo常用命令

- hexo g(generate)根据md文件生成网页
- hexo s(start) 运行nodejs，此时通过网页查看效果。
- pm2 restart all 重新启动所有pm2管理的项目
- pm2 status 查看状态
- hexo new "XXX" 新建文章
- hexo clean 如果生成失败需要清理之后再重新生成

## 附录

- `themes\hexo-theme-matery\layout\_partial\bg-cover-content.ejs`目录中指定了滑动图片的内容
