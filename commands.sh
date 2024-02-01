
sbatch --mem=200G \
       --gres=gpu:1 \
       --cpus-per-task=20 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Single_Agent_minigrid/tasks/jade1_RouteWithTrod.sh


sbatch --mem=200G \
       --gres=gpu:1 \
       --cpus-per-task=20 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Single_Agent_minigrid/tasks/jade2_RouteWithTrod.sh

