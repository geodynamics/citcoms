This container hosts a built version of citcoms.

docker run -it --rm -v $HOME/citcoms:/home/citcoms_user/work geodynamics/citcoms

This command will start the citcoms docker image and give you terminal access. Any changes made in the /home/citcoms_user/work directory will be reflected on the host machine at home/citcoms.
