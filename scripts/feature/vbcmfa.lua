local VBCMFA = {}

function VBCMFA:new(...)
   local o = {}
   setmetatable(o, self)
   self.__index = self
   o:__init(...)
   return o
end

function VBCMFA:__init(cfg)
   local n, s, p, k, f = self:_setandgetDims(cfg)

   self.hidden = {
      Zm = torch.Tensor(s, k, n),
      Zcov = torch.Tensor(s, k, k):zeros(),

      -- Xm is what is provided which is f x n for all s
      Xcov = torch.Tensor(s, f, f):zeros(),

      Qs = torch.Tensor(n, s),
      phim = torch.Tensor(s),

      get = function(self)
         return self._Xm, self._Xconv, self.Qs, self.am
      end
   }

   self.factorLoading = {
      Lm = torch.Tensor(s, p, k),
      Lcov = torch.Tensor(s, p, k, k),

      Gm = torch.Tensor(s, p, f),
      Gcov = torch.Tensor(s, p, f, f)

      a = 1,
      b = torch.Tensor(s, p),

      alpha = 1,
      beta = torch.Tensor(s, p),

      get = function(self)
         return self._Lm, self._Lcov, self.a, self.b
      end
   }

   self.hyper = {
      a_star = 1,
      b_star = 1,

      alpha_star = 1,
      beta_star = 1, 


      phi_star = 1,

      PsiI = torch.Tensor(p),

      get = function(self)
         return self.ustr, self.vstr, self.astr, self.bstr, self.alphastr, self.PsiI
      end
   }

   self:initparams()
end

function VBMFA:_setandgetDims(cfg)
   if cfg then
      self.n = cfg.batchSize
      self.s = cfg.numComponents
      self.p = cfg.outputVectorSize
      self.k = cfg.factorVectorSize
      self.f = cfg.inputVectorSize
   end
   return self.n, self.s, self.p, self.k, self.f
end

