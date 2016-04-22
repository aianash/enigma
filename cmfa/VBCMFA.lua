local pl = (require 'pl.import_into')()
require 'cephes'

----------------------
--[[ VBCMFA ]]--
-- Abstract class for Variational Bayesian Mixture of Factor Analyzer
-- y = Lz + Gx + e | s or Y = LZ + GX + e | S
-- L  - S x p x k
-- z  - S x k
-- Z  - S x k x n
-- G  - S x p x f
-- x  - S x f
-- X  - S x f x n
-- S  - n x S
--
-- Priors
-- p(L | 0, nu)
-- p(nu | a_star, b_star)
-- p(omega | alpha_star, beta_star)
-- p(s | pi)
-- p(pi | phi_star)
-- p(e | 0, Psi)
--
-- Posteriors
-- q(L | Lm, Lcov)
-- q(Z | Zm, Zcov, S)
-- q(nu | a, b)
--
-- q(G | Gm, Gcov)
-- q(X | Xm, Xcov, S)
-- q(omega | alpha, beta)
--
-- q(s)
-- q(pi | phim)
--
-- Hidden variables
-- Zm   - S x k x n            (hidden params)
-- Zcov - S x k x k            (hidden params)
--
-- Xm   - S x f x n
-- Xcov - S x f x f
--
-- Qs   - n x S
-- phim - S
--
-- Factor Loading parameters
-- Lm    - S x p x k             (each s component, each row p, k dimension of mean of Lambda)
-- Lcov  - S x p x k x k         (each s component, each row, p, kxk - dimensional cov matrix)
-- a     - 1
-- b     - S x k
--
-- Gm    - S x p x f
-- Gcov  - S x p x f x f
-- alpha - 1
-- beta  - S x f
--
-- Hyper parameters
-- a_star     - 1                        (hyper parameters for priori
-- b_star     - 1                        on
--
-- alpha_star - 1
-- beta_star  - 1
--
-- E_starI    - f x f
--
-- phi_star   - 1                        (a number)
--
-- PsiI       - p                        (a diagonal matrix)
----------------------
local VBCMFA = klazz('enigma.cmfa.VBCMFA')

--
function VBCMFA:__init(cfg)
   local n, S, p, k, f, N = self:_setandgetDims(cfg)

   self.hidden = {
      Zm = torch.randn(S, k, n),
      Zcov = torch.eye(k, k):repeatTensor(S, 1, 1),

      Qs = torch.ones(n, S) / S,
      phim = torch.ones(S) / S,

      getZ = function(self)
         return self.Zm, self.Zcov
      end,

      getS = function(self)
         return self.Qs, self.phim
      end
   }

   self.conditional = {
      Xm = torch.randn(S, f, n),
      Xcov = torch.eye(f, f):repeatTensor(S, 1, 1),

      getX = function(self)
         return self.Xm, self.Xcov
      end
   }

   self.factorLoading = {
      Lm = torch.randn(S, p, k),
      Lcov = torch.eye(k, k):repeatTensor(S, p, 1, 1),

      a = 1,
      b = torch.randn(S, k),

      Gm = torch.randn(S, p, f),
      Gcov = torch.eye(f, f):repeatTensor(S, p, 1, 1),

      alpha = 1,
      beta = torch.randn(S, f),

      getL = function(self)
         return self.Lm, self.Lcov, self.a, self.b
      end,

      getG = function(self)
         return self.Gm, self.Gcov, self.alpha, self.beta
      end
   }

   self.hyper = {
      a_star = 1,
      b_star = 1,

      alpha_star = 1,
      beta_star = 1,

      E_starI = torch.eye(f, f),

      phi_star = 1,

      PsiI = torch.ones(p),

      get = function(self)
         return self.a_star, self.b_star, self.alpha_star, self.beta_star, self.E_starI, self.phi_star, self.PsiI
      end
   }

   self.removal = cfg.removal

   self.batchIdx = -1
   self.tau = cfg.delay
   self.kappa = cfg.forgettingRate
   self.Fmatrix = torch.zeros(9, S)
   self.F = - 1 / 0
   self._rho = 0.5

   self.verbose = false
end

--
function VBCMFA:reset()
   local n, S, p, k, f = self:_setandgetDims()

   print("Resetting parameters")
   self.hidden.Zm = torch.randn(S, k, n)
   self.hidden.Zcov = torch.eye(k, k):repeatTensor(S, 1, 1)

   self.hidden.Qs = torch.ones(n, S) / S
   self.hidden.phim = torch.ones(S) / S

   self.conditional.Xm = torch.randn(S, f, n)
   self.conditional.Xcov = torch.eye(f, f):repeatTensor(S, 1, 1)

   self.factorLoading.Lm = torch.randn(S, p, k)
   self.factorLoading.Lcov = torch.eye(k, k):repeatTensor(S, p, 1, 1)

   self.factorLoading.a = 1
   self.factorLoading.b = torch.randn(S, k)

   self.factorLoading.Gm = torch.randn(S, p, f)
   self.factorLoading.Gcov = torch.eye(f, f):repeatTensor(S, p, 1, 1)

   self.factorLoading.alpha = 1
   self.factorLoading.beta = torch.randn(S, f)

   self.hyper.a_star = 1
   self.hyper.b_star = 1
   self.hyper.alpha_star = 1
   self.hyper.beta_star = 1
   self.hyper.E_starI = torch.eye(f, f)
   self.hyper.phi_star = 1
   self.hyper.PsiI = torch.ones(p)
end

--
function VBCMFA:rho(batchIdx)
   if not batchIdx then
      if self._rho then return self._rho
      elseif self.batchIdx == 1 then return 1
      else return (self.batchIdx + self.tau) ^ -self.kappa end
   else self.batchIdx = batchIdx end
end

--
function VBCMFA:_setandgetDims(cfg)
   if cfg then
      self.n = cfg.batchSize
      self.S = cfg.numComponents
      self.p = cfg.outputVectorSize
      self.k = cfg.factorVectorSize
      self.f = cfg.inputVectorSize
      self.N = cfg.datasetSize
   end
   return self.n, self.S, self.p, self.k, self.f, self.N
end

--
function VBCMFA:_checkDimensions(tensor, ...)
   local dimensions = {...}
   if #dimensions ~= tensor:nDimension() then
      return false, string.format("Wrong number of dimension = %d, expecting %d", tensor:nDimension(), #dimensions)
   end

   local res = true
   for idx, size in ipairs(dimensions) do
      res = res and (tensor:size(idx) == size)
      if not res then
         return false, string.format("wrong size = %d at dimension %d, expecting size = %d", tensor:size(idx), idx, size)
      end
   end

   return true
end

function VBCMFA:inferQL( ... ) error(FunctionNotImplemented("inferQL")) end
function VBCMFA:inferQZ( ... ) error(FunctionNotImplemented("inferQZ")) end
function VBCMFA:inferQnu( ... ) error(FunctionNotImplemented("inferQnu")) end
function VBCMFA:inferQG( ... ) error(FunctionNotImplemented("inferQG")) end
function VBCMFA:inferQX( ... ) error(FunctionNotImplemented("inferQX")) end
function VBCMFA:inferQomega( ... ) error(FunctionNotImplemented("inferQomega")) end
function VBCMFA:inferQs( ... ) error(FunctionNotImplemented("inferQs")) end
function VBCMFA:inferQpi( ... ) error(FunctionNotImplemented("inferQpi")) end
function VBCMFA:inferPsiI( ... ) error(FunctionNotImplemented("inferPsiI")) end
function VBCMFA:inferab( ... ) error(FunctionNotImplemented("inferab")) end
function VBCMFA:inferalphabeta( ... ) error(FunctionNotImplemented("inferalphabeta")) end
function VBCMFA:inferPhi( ... ) error(FunctionNotImplemented("inferPhi")) end
function VBCMFA:inferE_starI( ... ) error(FunctionNotImplemented("inferE_statI")) end
function VBCMFA:calcF( ... ) error(FunctionNotImplemented("calcF")) end
function VBCMFA:doBirth( ... ) error(FunctionNotImplemented("doBirth")) end

function VBCMFA:batchtrain( ... ) error(FunctionNotImplemented("batchtrain")) end
function VBCMFA:train( ... ) error(FunctionNotImplemented("train")) end

--
function VBCMFA:infer(tr, pause, ...)
   local c = os.clock()
   if self.verbose then print(string.format("\n== Infer %s ==", tr)) end
   self["infer"..tr](self, ...)
   if self.verbose then
      print(string.format("%s\t= %f", tr, os.clock() - c))
      print("------------------------------------------")
   end
   if pause then io.read() end
end

--
function VBCMFA:pr(tr, ns, pause)
   print(tr .. " Prev")
   print(self.old[tr])

   print(tr .. " New")
   print(self[ns][tr])

   if pause then io.read() end
end

--
function VBCMFA:check(X, name)
   if X:squeeze():nDimension() == 1 then
      return
   end
   local nDim = X:nDimension()
   local sizes = torch.Tensor(torch.Storage(nDim):copy(X:size()))
   local X1 = X:view(sizes:prod(), 1)

   local v, indices = torch.max(X1:ne(X1), 1)
   v = v:squeeze()

   if v == 1 then
      for i = 1, indices:size(1) do
         local lidx = indices[i]:squeeze()
         local idx = {}
         for d = 1, nDim - 1 do
            local sz = sizes:narrow(1, d + 1, nDim - d):prod()
            idx[d] = math.floor(lidx / sz + 1)
            lidx = lidx - (idx[d] - 1) * sz
         end
         idx[nDim] = math.floor(lidx)
         local str = name
         local x = X
         for _, i in ipairs(idx) do
            str = str.."["..tostring(i).."]"
            x = x[i]
         end
         print(string.format(str.." = %f", x))
      end
      os.exit()
   end
end

return VBCMFA