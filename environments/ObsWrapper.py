from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper


class FullRGBImgPartialObsWrapper(RGBImgPartialObsWrapper):
    def observation(self, obs):
        rgb_img_partial = self.get_frame(tile_size=self.tile_size, agent_pov=False)

        return {**obs, "image": rgb_img_partial}
