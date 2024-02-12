import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import { useLoginMutation } from "../Slices/usersApiSlice";
import { setCredentials } from "../Slices/authSlices";
import { toast } from "react-toastify";
import Loader from "../Components/Loader";

const LoginPage = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const navigate = useNavigate();
  const dispatch = useDispatch();

  const [login, { isLoading }] = useLoginMutation();

  const { userInfo } = useSelector((state) => state.auth);

  useEffect(() => {
    if (userInfo) {
      navigate("/");
    }
  }, [navigate, userInfo]);

  const submitHandler = async (e) => {
    e.preventDefault();
    try {
      const res = await login({ email, password }).unwrap(); // this returns a promise so we unwrap it
      dispatch(setCredentials({ ...res }));
      navigate("/");
    } catch (error) {
      toast.error(error?.data?.message || error.error);
    }
  };

  return (
    <div className="flex justify-center items-center flex-col min-h-[var(--min-page-height)] font-bold">
      <div className="max-w-xs w-3/4 border rounded-xl p-4">
        <h1 className="text-4xl mb-8">Sign In</h1>
        <form
          action=""
          onSubmit={submitHandler}
          className="flex flex-col gap-2  font-semibold"
        >
          <label htmlFor="email">Email</label>
          <input
            required
            name="email"
            id="email"
            type="text"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Type here"
            className="input input-bordered w-full max-w-xs"
          />
          <label htmlFor="password">Password</label>
          <input
            name="password"
            required
            id="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            type="password"
            placeholder="Type here"
            className="input input-bordered w-full max-w-xs"
          />
          {isLoading ? (
            <button className="btn btn-primary w-[6rem] mt-2" type="submit">
              <Loader></Loader>
            </button>
          ) : (
            <button className="btn btn-primary w-[6rem] mt-2" type="submit">
              Login!
            </button>
          )}
        </form>
        <h4 className="font-semibold text-sm mt-4">
          New account? Register{" "}
          <Link className="text-primary" to={"/register"}>
            here.
          </Link>
        </h4>
      </div>
    </div>
  );
};
export default LoginPage;
