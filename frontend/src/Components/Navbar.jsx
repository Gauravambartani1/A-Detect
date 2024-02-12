import { Link, useNavigate } from "react-router-dom";
import ThemeSwitcher from "./ThemeSwitcher.jsx";
import { useSelector, useDispatch } from "react-redux";
import { useLogoutMutation } from "../Slices/usersApiSlice.js";
import { clearCredentials } from "../Slices/authSlices.js";

const Navbar = () => {
	const { userInfo } = useSelector((state) => state.auth);
	const [logout] = useLogoutMutation();

	const dispatch = useDispatch();
	const navigate = useNavigate();

	const logoutHandler = async () => {
		try {
			await logout().unwrap();
			dispatch(clearCredentials());
			navigate("/");
		} catch (error) {
			console.log("error");
		}
	};

	return (
		<div className="navbar bg-base-100 font-semibold">
			<div className="flex-1">
				<Link
					className="btn btn-ghost normal-case text-xl font-bold"
					to={"/"}
				>
					🧠A-Detect
				</Link>
			</div>
			<div className="flex-none">
				<ul className="menu menu-horizontal px-1 space-x-2">
					<li className="mt-1 ">
						<ThemeSwitcher />
					</li>
					{userInfo ? (
						<>
							<li>
								<details>
									<summary>{userInfo.name}</summary>
									<ul className="p-2 bg-base-100">
										<li>
											<Link to={"/profile"}>Profile</Link>
										</li>
										<li>
											<a onClick={logoutHandler}>
												Logout
											</a>
										</li>
									</ul>
								</details>
							</li>
						</>
					) : (
						<>
							<li>
								<Link to={"/login"}>Login</Link>
							</li>
							<li>
								<Link to={"/register"}>Sign Up</Link>
							</li>
						</>
					)}
				</ul>
			</div>
		</div>
	);
};

export default Navbar;
