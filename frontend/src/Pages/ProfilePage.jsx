import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import { useUpdateMutation } from "../Slices/usersApiSlice";
import { setCredentials } from "../Slices/authSlices";
import { toast } from "react-toastify";
import Loader from "../Components/Loader";

const ProfilePage = () => {
  const { userInfo } = useSelector((state) => state.auth);

  const [email, setEmail] = useState(userInfo.email);
  const [name, setName] = useState(userInfo.name);
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  const dispatch = useDispatch();

  const [update, { isLoading }] = useUpdateMutation();

  const submitHandler = async (e) => {
    e.preventDefault();
    if (password !== confirmPassword) {
      toast.error("Passwords do not match");
    } else {
      try {
        const res = await update({ name, email, password }).unwrap(); // this returns a promise so we unwrap it
        dispatch(setCredentials({ ...res }));
        toast.success("Credentials changed successfully.");
      } catch (error) {
        toast.error(error?.data?.message || error.error);
      }
    }
  };

  return (
    <div className="flex justify-center items-center flex-col min-h-[var(--min-page-height)] font-bold">
      <div className=" max-w-xs w-3/4 border rounded-xl p-4">
        <h1 className="text-4xl mb-8">Change credentials.</h1>
        <form
          action=""
          onSubmit={submitHandler}
          className="flex flex-col gap-2  font-semibold"
        >
          <label htmlFor="name">Name</label>
          <input
            name="name"
            id="name"
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="enter name"
            className="input input-bordered w-full max-w-xs"
          />
          <label htmlFor="email">Email</label>
          <input
            name="email"
            id="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            type="text"
            placeholder="enter email"
            className="input input-bordered w-full max-w-xs"
          />
          <label htmlFor="password">Password</label>
          <input
            name="password"
            id="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            type="password"
            placeholder="enter new password"
            className="input input-bordered w-full max-w-xs"
          />
          <label htmlFor="confirm" className="text-secondary">
            Confirm Password
          </label>
          <input
            name="confirm"
            id="confirm"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            type="password"
            placeholder="confirm new password"
            className="input input-bordered w-full max-w-xs"
          />
          {isLoading ? (
            <button className="btn btn-primary w-[6rem] mt-2" type="submit">
              <Loader></Loader>
            </button>
          ) : (
            <button className="btn btn-primary w-[6rem] mt-2" type="submit">
              Change
            </button>
          )}
        </form>
      </div>
    </div>
  );
};
export default ProfilePage;
