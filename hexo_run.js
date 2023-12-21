exec('hexo server',(error, stdout, stderr) => {
    if(error){
            console.log('exec error: ${error}')
            return
    }
    console.log('stdout: ${stdout}');
    console.log('stderr: ${stderr}');
})