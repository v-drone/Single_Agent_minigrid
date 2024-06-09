@echo off
setlocal enabledelayedexpansion

REM 设置镜像 ID
set "imageID=25efb285a557"

REM 循环创建容器和映射卷
for /L %%i in (0, 1, 400) do (
    REM 计算容器编号
    set /a "containerNum=5000 + %%i"

    REM 设置容器名
    set "containerName=!containerNum!"

    REM 设置卷映射路径
    set "hostPath=C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid\frpc\frpc_%%containerNum.toml"
    set "containerPath=/etc/frp/frpc.toml"

    REM 创建 Docker 容器并映射卷
    docker run -d --name !containerName! -v "!hostPath!:!containerPath!" !imageID!

    REM 输出创建容器信息
    echo Created container !containerName! with volume from !hostPath! to !containerPath!
)

endlocal
